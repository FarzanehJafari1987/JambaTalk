import numpy as np
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor

from data_loader import get_dataloaders
from jambatalk import JambaTalk


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, save_path, epoch):
    os.makedirs(save_path, exist_ok=True)
    ckpt_file = os.path.join(save_path, f"{epoch}_model.pth")
    torch.save(model.state_dict(), ckpt_file)
    print(f"[INFO] Saved checkpoint: {ckpt_file}")


def trainer(args, train_loader, dev_loader, model, optimizer, criterion):
    timestamp = time.strftime("%m_%d_%H_%M", time.localtime())
    save_path = os.path.join(args.dataset, f"{args.save_path}_{args.feature_dim}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    # Resume training if requested
    if args.last_train > 0 and args.load_path:
        ckpt_file = os.path.join(args.load_path, f"{args.last_train}_model.pth")
        print(f"[INFO] Loading checkpoint from {ckpt_file}")
        model.load_state_dict(torch.load(ckpt_file, map_location=args.device))

    writer = SummaryWriter(log_dir=f"log/{os.path.basename(save_path)}")
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

    iteration = 0
    for epoch in range(1, args.max_epoch + 1):
        # Training
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        optimizer.zero_grad()

        for i, (audio, vertice, template, _) in pbar:
            iteration += 1
            audio, vertice, template = audio.to(args.device), vertice.to(args.device), template.to(args.device)
            vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits = model(audio, template, vertice)

            if args.dataset == 'BIWI':
                vertice = vertice.reshape(1, -1, vertice.size()[2] * vertice.size()[3])

            # align vertices
            min_len = min(vertice_out.shape[1], vertice.shape[1])
            vertice_out_aligned = vertice_out[:, :min_len]
            vertice_aligned = vertice[:, :min_len]

            loss1 = criterion(vertice_out_aligned, vertice_aligned)

            pred_vel = vertice_out_aligned[:, 1:, :] - vertice_out_aligned[:, :-1, :]
            gt_vel = vertice_aligned[:, 1:, :] - vertice_aligned[:, :-1, :]
            loss2 = criterion(pred_vel, gt_vel)
            
            loss3 = criterion(lip_features, text_hidden_states)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            text_logits = torch.argmax(text_logits, dim=-1)
            text_logits = processor.batch_decode(text_logits)
            text_logits = tokenizer(text_logits, return_tensors="pt").input_ids.to(args.device)
            loss4 = nn.functional.ctc_loss(
                log_probs,
                text_logits,
                torch.tensor([log_probs.shape[0]], device=args.device),
                torch.tensor([text_logits.shape[1]], device=args.device),
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )

            loss = torch.mean(1000 * loss1 + 1000 * loss2 + 0.001 * loss3 + 0.0001 * loss4)
            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": loss.item(), "loss1": loss1.item(), "loss2": loss2.item(), "loss3": loss3.item(), "loss4": loss4.item()})

        writer.add_scalar("train/loss", loss.item(), epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)

        # Validation
        model.eval()
        valid_loss_log = []
        with torch.no_grad():
            for audio, vertice, template, _ in dev_loader:
                audio, vertice, template = audio.to(args.device), vertice.to(args.device), template.to(args.device)
                vertice_out, vertice, *_ = model(audio, template, vertice)
                if args.dataset == 'BIWI':
                    vertice = vertice.reshape(1, -1, vertice.size()[2] * vertice.size()[3])
                min_len = min(vertice_out.shape[1], vertice.shape[1])
                loss_val = criterion(vertice_out[:, :min_len], vertice[:, :min_len])
                valid_loss_log.append(loss_val.item())

        mean_val_loss = np.mean(valid_loss_log)
        writer.add_scalar("val/loss", mean_val_loss, epoch)

        print(f"[Epoch {epoch}] Train Loss: {loss.item():.9f}, Val Loss: {mean_val_loss:.9f}")

        if epoch % 100 == 0 or epoch == args.max_epoch:
            save_checkpoint(model, save_path, epoch)

    writer.close()


def test(args, model, test_loader, epoch):
    timestamp = time.strftime("%m_%d_%H_%M", time.localtime())
    result_path = os.path.join(args.dataset, f"{args.result_path}_{timestamp}")
    os.makedirs(result_path, exist_ok=True)

    save_path = os.path.join(args.dataset, args.save_path)
    ckpt_file = os.path.join(save_path, f"{epoch}_model.pth")
    print(f"[INFO] Loading checkpoint from {ckpt_file}")
    model.load_state_dict(torch.load(ckpt_file, map_location=args.device))
    model.to(args.device).eval()

    train_subjects_list = args.train_subjects.split()

    for audio, vertice, template, file_name in test_loader:
        audio, vertice, template = audio.to(args.device), vertice.to(args.device), template.to(args.device)
        test_subject = "_".join(file_name[0].split("_")[:-1])

        subjects_to_try = [test_subject] if test_subject in train_subjects_list else train_subjects_list

        for condition_subject in subjects_to_try:
            prediction, lip_features, logits = model.predict(audio, template)
            prediction = prediction.squeeze()
            out_file = os.path.join(result_path, f"{file_name[0].split('.')[0]}_condition_{condition_subject}.npy")
            np.save(out_file, prediction.detach().cpu().numpy())

    print(f"[INFO] Test results saved in {result_path}")


def main():
    parser = argparse.ArgumentParser(description='JambaTalk: Speech-driven 3D Talking Head Generation based on Hybrid Transformer-Mamba Model')
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dataset", type=str, default="BIWI")
    parser.add_argument("--vertice_dim", type=int, default=3895 * 3)
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--period", type=int, default=25)
    parser.add_argument("--wav_path", type=str, default="wav")
    parser.add_argument("--vertices_path", type=str, default="vertices_npy")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl")
    parser.add_argument("--save_path", type=str, default="save")
    parser.add_argument("--result_path", type=str, default="result")
    parser.add_argument("--last_train", type=int, default=0)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--val_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")
    args = parser.parse_args()


    # parser = argparse.ArgumentParser(description='JambaTalk: Speech-driven 3D Talking Head Generation based on Hybrid Transformer-Mamba Model')
    # parser.add_argument("--mode", type=str, choices=["train", "test"], default="test")
    # parser.add_argument("--lr", type=float, default=1e-4)
    # parser.add_argument("--dataset", type=str, default="vocaset")
    # parser.add_argument("--vertice_dim", type=int, default=5023 * 3)
    # parser.add_argument("--feature_dim", type=int, default=512)
    # parser.add_argument("--period", type=int, default=30)
    # parser.add_argument("--wav_path", type=str, default="wav")
    # parser.add_argument("--vertices_path", type=str, default="vertices_npy")
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # parser.add_argument("--max_epoch", type=int, default=100)
    # parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--template_file", type=str, default="trainingdata/templates.pkl")
    # parser.add_argument("--save_path", type=str, default="save_512_09_04_08_35")
    # parser.add_argument("--result_path", type=str, default="result")
    # parser.add_argument("--last_train", type=int, default=0)
    # parser.add_argument("--load_path", type=str, default=None)
    # parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
    #                                                           " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
    #                                                           " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
    #                                                           " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    # parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
    #                                                         " FaceTalk_170908_03277_TA")
    # parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
    #                                                          " FaceTalk_170731_00024_TA")
    # args = parser.parse_args()

    # Resolve device
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model
    model = JambaTalk(args)
    print("Model parameters:", count_parameters(model))
    model = model.to(args.device)

    # Load data
    dataset = get_dataloaders(args)

    if args.mode == "train":
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        trainer(args, dataset["train"], dataset["valid"], model, optimizer, criterion)

    elif args.mode == "test":
        test(args, model, dataset["test"], epoch=args.max_epoch)


if __name__ == "__main__":
    main()
