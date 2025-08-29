# JambaTalk
![Jambatalk release](./media/jamba_logo.png)

# JambaTalk: Talking Face Generation Framework

Official PyTorch implementation for **JambaTalk**, a framework for generating realistic 3D talking faces.

---

<p align="center">
<img src="./media/JambaTalk.png" width="90%" />
</p>

> Given a speech signal as input, our framework can generate realistic 3D talking faces that synchronize with audio and preserve comprehensibility through lip-reading alignment.

---

## **Environment Requirements**

- Linux or macOS (Windows not fully tested)
- Python 3.8+
- PyTorch 1.12.1
- CUDA 11.3+
- ffmpeg
- [MPI-IS/mesh](https://github.com/MPI-IS/mesh)

### Setup

Clone the repo:
```bash
git clone https://github.com/jambatalk/JambaTalk_release.git
cd JambaTalk_release
```

Create and activate conda environment:
```bash
conda create -n jambatalk python=3.8
conda activate jambatalk
```

Install dependencies:
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

---

## **Datasets**

### VOCASET
1. Request VOCASET from [https://voca.is.tue.mpg.de/](https://voca.is.tue.mpg.de/).  
2. Place files `data_verts.npy`, `raw_audio_fixed.pkl`, `templates.pkl`, and `subj_seq_to_idx.pkl` into `vocaset/`.  
3. Download `FLAME_sample.ply` from [VOCA template](https://github.com/TimoBolkart/voca/tree/master/template) and put it in `vocaset/`.  
4. Process the data:  
   ```bash
   cd vocaset
   python process_voca_data.py
   ```

### BIWI
Follow instructions in [`BIWI/README.md`](BIWI/README.md) to preprocess BIWI. Place `.npy` and `.wav` files into `BIWI/vertices_npy` and `BIWI/wav`.

---

## **Demo**

Download pretrained models:  
- [BIWI.pth](https://drive.google.com/file/d/1ZGdEVcLa3W0SLMGXOYOJlxikHALhqQ7s/view?usp=sharing)  
- [VOCASET.pth](https://drive.google.com/file/d/1iwxw4snYndoip2u2Iwe7h-rfPhVJRm2U/view?usp=sharing)  

Run demo with your audio:
```bash
python demo_voca.py --wav_path "demo/wav/test.wav" --subject FaceTalk_170908_03277_TA
python demo_BIWI.py --wav_path "demo/wav/test.wav" --subject M1
```

Results will be saved in `demo/output/`.

---

## **Training and Testing**

We provide training and testing scripts for both **VOCASET** and **BIWI** datasets.  
See detailed commands in [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md).

---

## **Using Your Own Dataset**

1. Create `<dataset_dir>` inside the project.  
2. Place `.npy` vertices into `<dataset_dir>/vertices_npy` and `.wav` audio files into `<dataset_dir>/wav`.  
3. Save subject templates into `<dataset_dir>/templates.pkl` and at least one `.ply` template in `<dataset_dir>/templates/`.  
4. Train with:
   ```bash
   python main.py --dataset <dataset_dir> --vertice_dim <num_vertices*3>
   ```

---

## **Citation**
If you use JambaTalk in research, please cite:

```
@misc{jambatalk2023,
  title={JambaTalk: Talking Face Generation Framework},
  author={Jambatalk Research Team},
  year={2023},
  note={https://github.com/jambatalk/JambaTalk_release}
}
```

---

## **Acknowledgements**
This project builds on ideas and resources from:
- [FaceFormer](https://github.com/EvelynFan/FaceFormer)
- [CodeTalker](https://github.com/Doubiiu/CodeTalker)
- [FaceXHuBERT](https://github.com/galib360/FaceXHuBERT)
- [VOCASET](https://voca.is.tue.mpg.de/) & [B3D(AC)2](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html)
- [MPI-IS/mesh](https://github.com/MPI-IS/mesh)

---

## **Contact**
- 📧 Research inquiries: research@jambatalk.com  
- 💼 Business licensing: licensing@jambatalk.com  

---

## **License**
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See [LICENSE](LICENSE).

---

## **Join Us**
At **Jambatalk**, we are building the future of conversational AI avatars.  
If you are passionate about AI, speech, and 3D avatars, join us to shape the next generation of digital human technology.  

🌐 [Visit Jambatalk](https://www.jambatalk.com) | 📩 careers@jambatalk.com
