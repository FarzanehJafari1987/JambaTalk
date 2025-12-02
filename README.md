# JambaTalk: Speech-driven 3D Talking Head Generation based on a Hybrid Transformer-Mamba Model
## ACM Transactions on Multimedia Computing, Communications, and Applications
Farzaneh Jafari, Stefano Berretti, Anup Basu

[[Paper]](https://arxiv.org/pdf/2408.01627)|[[Project Page]](https://farzanehjafari1987.github.io/JambaTalk.github.io/)|[[License]](https://github.com/FarzanehJafari1987/JambaTalk/blob/main/LICENSE)

![Jambatalk release](./media/JambaTalk.png)
The Wav2Vec 2.0 model is used to extract features from the input speech,
with the encoder initialized using pre-trained weights from the original model [5]. These encoded features are
passed to the JambaTalk decoder, which generates a sequence of animated 3D face meshes. The Transformer
layer incorporates Low-Rank Learned Rotary Positional Embedding (LRL-RoPE) and Grouped-Query Attention
(GQA), providing a computation-efficient alternative to traditional Transformers. The lip feature extraction
block then converts motion decoder outputs into lip deformation features by selecting lip vertices with a lip
mask, which are processed by a Transformer-based lip encoder in the lip reader module to synchronize lip
shapes.

---

## **Environment Requirements**

- Linux or macOS (Windows not fully tested)
- Python 3.9+
- PyTorch 2.4.1
- CUDA 11.8
- ffmpeg
- mamba_ssm
- [MPI-IS/mesh](https://github.com/MPI-IS/mesh)

## Setup

Clone the repository:
```bash
git clone https://github.com/jambatalk/JambaTalk_release.git
cd JambaTalk_release
```
Create and activate a conda environment:
```bash
conda create -n jambatalk python=3.9
conda activate jambatalk
```

Install dependencies:
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
---

## **Dataset Requirements**
### VOCASET Configuration
Obtain the VOCASET dataset from [https://voca.is.tue.mpg.de/](https://voca.is.tue.mpg.de/). After downloading, organize the files `data_verts.npy`, `raw_audio_fixed.pkl`, `templates.pkl`, and `subj_seq_to_idx.pkl` within the `vocaset/` directory. Additionally, retrieve "FLAME_sample.ply" from the [voca repository](https://github.com/TimoBolkart/voca/tree/master/template) and store it in `vocaset/templates`. Read the vertices/audio data and convert them to .npy/.wav files stored in `vocaset/vertices_npy` and `vocaset/wav`:
```
cd vocaset
python process_voca_data.py
```

### BIWI Setup
Acquire the BIWI dataset from the [Biwi 3D Audiovisual Corpus of Affective Communication](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html). This collection includes three main components:

- Binary facial geometry files (.vl format) located in the 'faces' directory
- Template mesh files (.obj format) found in 'rigid_scans'
- Audio recordings (.wav format) stored in the 'audio' folder

Organize by placing 'faces' and 'rigid_scans' directories in `BIWI`, while moving all audio files to `BIWI/wav`.
Process the geometric mesh data and transform it into `.npy` format for storage in `BIWI/vertices_npy`.

---

## **Running the Demo**

First, obtain the pre-trained model weights:
- [biwi.pth](https://drive.google.com/drive/folders/1wEOe26OWXnvlRiNxSZvYoHLYsNP1bmXR)  
- [vocaset.pth](https://drive.google.com/drive/folders/1LAgtaTZ9yjY0h26ozhWP8hA4wj422aeh)  

For BIWI mesh animation, execute the following command to generate facial animations using BIWI topology:
```
python demo_BIWI.py --wav_path "demo/wav/test.wav" --subject M1
```
For FLAME topology animation, use this command to create animations with FLAME mesh structure:
```
python demo_voca.py --wav_path "demo/wav/test.wav" --subject FaceTalk_170908_03277_TA
```
The system will automatically create rendered video outputs within the demo/output directory. For custom testing, place your audio files (.wav format) in the demo/wav directory and update the --wav_path "demo/wav/test.wav" parameter to reference your specific file.

---

## **VOCASET Model Training and Evaluation**

## Model Training and Evaluation
Execute the complete training pipeline on VOCASET and generate evaluation metrics using the test dataset:
```bash
python main.py --dataset vocaset --vertice_dim 15069 --feature_dim 512 --period 30 --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" --val_subjects "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA"
```
Output files, including evaluation metrics and trained model checkpoints, will be stored in `vocaset/result` and `vocaset/save`, respectively.

## Result Rendering
Generate visual outputs from the trained model:
```bash
python render.py --dataset vocaset --vertice_dim 15069 --fps 30
```
The generated visualizations will be available in the `vocaset/output` directory.

---

## **BIWI Model Training and Evaluation**

## Model Training and Assessment
Execute the training workflow on the BIWI dataset and generate performance results using the evaluation subset:
```bash
python main.py --dataset BIWI --vertice_dim 11685 --feature_dim 1024 --period 25 --train_subjects "F2 F3 F4 M3 M4 M5" --val_subjects "F2 F3 F4 M3 M4 M5" --test_subjects "F1 F5 F6 F7 F8 M1 M2 M6"
```
Performance metrics will be stored in the `BIWI/result` directory, while model checkpoints will be preserved in the `BIWI/save` directory.

## Output Visualization
Create visual representations of the model predictions:
```bash
python render.py --dataset BIWI --vertice_dim 11685 --fps 25
```
Generated video content will be accessible in the `BIWI/output` directory.

---

### **Custom Dataset Integration**

Establish a `<dataset_dir>` folder within the project structure.
Organize your data files by placing vertex data (.npy format) in `<dataset_dir>/vertices_npy` and audio recordings (.wav format) in `<dataset_dir>/wav`.
Store subject template information in `<dataset_dir>/templates.pkl` and include at least one mesh template (.ply format) within `<dataset_dir>/templates/`.
Launch the training process using:
```bash
python main.py --dataset <dataset_dir> --vertice_dim <num_vertices*3>
```

---

## **Citation**
If you use JambaTalk in research, please cite:

```
@misc{jambatalk2024Jafari,
  title={JambaTalk: Speech-driven 3D Talking Head Generation based on a Hybrid Transformer-Mamba Model},
  author={Farzaneh Jafari, Stefano Berretti, Anup Basu},
  year={2024},
  note={arXiv preprint arXiv:2408.01627}
}
```
## **Acknowledgements**
This project builds on ideas and resources from:
- [FaceFormer](https://github.com/EvelynFan/FaceFormer)
- [CodeTalker](https://github.com/Doubiiu/CodeTalker)
- [FaceDiffuser](https://github.com/uuembodiedsocialai/FaceDiffuser)
- [SelfTalk](https://github.com/psyai-net/SelfTalk_release/tree/main)
- [FaceXHuBERT](https://github.com/galib360/FaceXHuBERT)
- [VOCASET](https://voca.is.tue.mpg.de/)
- [BIWI](https://github.com/Doubiiu/CodeTalker/blob/main/BIWI/README.md)
- [MPI-IS/mesh](https://github.com/MPI-IS/mesh)
- [Mamba](https://github.com/state-spaces/mamba)
- [Jamba](https://github.com/kyegomez/Jamba)

---

## **License**
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See [LICENSE](LICENSE).
