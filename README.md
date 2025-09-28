# JambaTalk: Speech-driven 3D Talking Head Generation based on a Hybrid Transformer-Mamba Model
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

---

## **Running the Demo**

First, obtain the pre-trained model weights:
- [biwi.pth]()  
- [vocaset.pth]()  

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
