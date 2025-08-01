# Master-Exam

This project contains my Master Exam (planning, writing, ...)

- [Master-Exam](#master-exam)
    - [Project Details](#project-details)
    - [Plan](#plan)
    - [Writing](#writing)
    - [Source](#source)
    - [Data](#data)
    - [Image-to-Image Translation Methods](#image-to-image-translation-methods)
    - [Own Architecture](#own-architecture)
    - [Research](#research)
    - [Commands](#commands)
    - [Progress](#progress)
    - [Meetings](#meetings)
    - [Fragen](#fragen)
    - [Das sagt die KI](#das-sagt-die-ki)






---
### Project Details



**Title:** Can generative models learn complex relationships?



**Task:** (problem/research question)<br>The aim is to investigate whether generative models can generate physically correct data. Specifically, the generation of sound propagation in urban areas will be investigated.

- Input: Satellite image + temperatur + humidity + source power (dB)

- Output: Sound propagation (sound frequency for each pixel)

  

**Additional Conditions:**

- Python + Pytroch?

- ...

  

**Goals:**

- Analysis and evaluation of current generative approaches to map physical relationships
- Creation of a training dataset (enough data? Possibly extend with synthetic data?)
- Development of a benchmark metric to evaluate generative models
- Elaborate current generative approaches
- Implement current generative approaches
- Train or pre-train current generative approaches?
- Collaboration in the research project (?)
- Writing the Master's thesis



**Theses:** (sort out and adjust)

1. Generative models can predict the probabilities of complex physical processes based on an image
2. Physical Constraints and focuses are important/improve the accuracy of physical correct generations
3. PINNs are better capable to generate physical correct generations/images as classical/modern generative models/methods
4. ...



**Coorparation:**<br>The project is being carried out in cooperation with Herrenknecht Vertikal GmbH and the Institute for Machine Learning and Analytics (IMLA) as part of the BMBF-funded KI-Bohrer research project.

- https://www.ki-bohrer.de/
- https://imla.hs-offenburg.de/






---
### Plan

1. Find the exact topic of your master's degree (with research question and subtopics)
2. Research of this topic
3. Update topic + define thesis(es)
4. Set up experiment plan (how measurable?)
5. Set up experiments (possibly create architectures)
6. Perform experiment (train + result)
7. Start writing / writing plan
8. ...




---
### Writing

The latex code is on Overleaf and here the code is also updated without the images.

https://www.overleaf.com/project/679f743ec9a1ac78e5db574b

The HS-Offenburg template was used and uploaded to Overleaf.
On ‘Menu’ there are some settings and also options for exporting the code. I often export/download the code as a ZIP file, then extract it to the Latex folder and update the Latex folder in this Git project. The .gitignore file ensures that the PDF itself and the images are not tracked.

> It is important to set the **main document** to **thesis.tex** as overleaf will automatically find preamble.tex by default (as it sets the document class) -> however, this is wrong and will lead to endless compilation until memory runs out.


All other options are listed here:
- Compiler: pdfLaTeX
- TeX Live version: 2024
- Main document: thesis.tex
- Spell check: English (American)
- Auto-complete: On
- Auto-close Brackets: On
- Code check: On
- Equation preview: On
- Editor theme: textmate
- Overall theme: Default
- Keybindings: None
- Font Size: 12px
- Font Family: Lucida / Source Code Pro
- Line Height: Normal
- PDF Viewer: Overleaf






---
### Source

Here all important resources (most likely papers) are listed with the probably most helpful statements.

Topic is *Deterministic Physics-Based Supervised Image-to-Image Translation* or *Physics-Based Supervised One-to-One Image Translation for predicting sound propagation*.



Also see: [GitHub - ShenZheng2000/Awesome-Image-to-Image-Translation: This repository contains representative papers for image-to-image translation (I2IT).](https://github.com/ShenZheng2000/Awesome-Image-to-Image-Translation)

Papers are collected in Zotero and will be exported to bibtex + will be commented in a documentation. Some information/papers are also here, but it changed to zotero and the tex documentation.<br>
Papers to process (add to Zotero + little summary -> take aways):

- [x] https://arxiv.org/abs/1708.02002 ✅-> Weighted Sample Difficulty Loss (Classification)
- [x] https://www.sciencedirect.com/science/article/pii/S0306457322000504 ❌ -> Unsupervised image-to-image translation with a new loss, but should not be appliable for paired images/physical -> it goes away from the exact pixel wise loss, but for physics we want exact pixels (for style-transfer not)
- [x] https://www.arxiv.org/pdf/2409.20031 ❌ -> off topic (microphone sound source localization)
- [x] https://openreview.net/pdf?id=js2ssA77fX ✅-> Random Masking of input (&predictions/loss), to learn the locals and not the whole global and also learn the transfer with less context which increases the training speed and ability to generalize
- [x] https://proceedings.neurips.cc/paper/2021/hash/e0cd3f16f9e883ca91c2a4c24f47b3d9-Abstract.html ✅ -> similar idea (reduces noisy gradients with saliency maps)
- [x] https://ieeexplore.ieee.org/abstract/document/9465428 ✅ -> similar idea (creates a weights-map where sharper areas get heavier weighted and so the model can focus more on it, but only the abstract core principle is interesting but everything else seems off topic)
- [x] https://arxiv.org/abs/2308.00947 ✅ -> similar idea (creates Saliency Maps, which show interesting parts of the image)
- [x] https://www.sciencedirect.com/science/article/pii/S1566253522001683 ✅ -> similar idea (using saliency maps during training)
- [x] https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html ✅ -> Alternative Sampling (just showing more hard examples during training and find hard examples dynamically during training)
- [x] https://proceedings.mlr.press/v80/ren18a.html ✅ -> Alternative Sampling (just showing more hard examples during training and find hard examples dynamically during training)
- [x] https://openaccess.thecvf.com/content_iccv_2017/html/Dong_Class_Rectification_Hard_ICCV_2017_paper.html ❌ -> off topic (class imbalance fix)
- [x] https://dl.acm.org/doi/abs/10.1145/1553374.1553380 ✅ -> Alternative Sampling (like humans showing harder and harder examples)
- [x] https://link.springer.com/article/10.1007/s11263-022-01611-x ✅ -> similiar as paper before (easy to hard observations)
- [x] https://ieeexplore.ieee.org/abstract/document/9392296 ✅ -> similiar as paper before (easy to hard observations)
- [x] https://arxiv.org/abs/2308.08468 ✅ -> make loss physics informed
- [x] https://arxiv.org/abs/2205.06858 ✅ -> make loss physics informed
- [x] https://arxiv.org/pdf/1710.11431 ✅ -> make loss physics informed


- [x] https://arxiv.org/abs/1903.07291 ❌ -> off topic (spatial normalization for more details)
- [x] https://arxiv.org/abs/1611.07004 ✅ -> pix2pix one of the basic generative model architectures
- [x] https://arxiv.org/abs/2110.04627 ❌ -> off topic (NLP GAN with transformer)


- [x] https://arxiv.org/abs/2012.09841 ✅ ->  NLP GAN changed to Image GAN with indirect saliency maps
- [x] https://arxiv.org/abs/2309.12106 ❌ -> off topic (loss for shapes/segmentation with fourier domain -> we does not have much shapes and does not predict shapes)
- [x] https://openaccess.thecvf.com/content/ICCV2021/papers/Fuoli_Fourier_Space_Losses_for_Efficient_Perceptual_Image_Super-Resolution_ICCV_2021_paper.pdf ❌ -> off topic (loss for shapes/segmentation with fourier domain -> we does not have much shapes and does not predict shapes)
- [x] https://arxiv.org/abs/2007.08579 ❌ -> off topic
- [x] https://arxiv.org/abs/2006.09661 ✅ -> Sinus-Activation Function with simple NN layers, could be good used as Decoder (mapping from latent space to image)
- [x] https://arxiv.org/abs/2010.08895 ✅ -> Fourier Neural Operators (FNOs) are deep learning models that leverage Fourier transforms to efficiently solve partial differential equations by learning the mapping between input and output functions in the frequency domain
- [x] https://arxiv.org/abs/2111.05826 ✅ -> Diffusion Models Architecture but not really interesting for determinstic image-to-image translation
- [x] https://epubs.siam.org/doi/abs/10.1137/18M1225409 ✅ -> Physics Informed GAN -> not Image to Image
- [x] https://ieeexplore.ieee.org/abstract/document/8945120 ✅ -> Physics Informed GANs -> not Image to Image
- [x] https://arxiv.org/abs/2010.08103 ✅ -> Image to Image Physic Informed GAN -> Application to replace expensive simulation with generative models
- [x] https://www.nature.com/articles/s42254-021-00314-5 ❌ -> Not available (General Physics in Machine Learning)
- [x] https://research.ebsco.com/c/gizthx/search/details/4nx6hvvm6v?db=asx ❌ -> No access
- [x] https://hal.science/hal-02405722/ ❌ -> A physics based approach but too far away
- [x] https://link.springer.com/article/10.1007/s10668-024-05504-6 ❌ -> Off-topic
- [x] https://arxiv.org/abs/2101.11986 ❌ -> Vision Transformer
- [x] https://arxiv.org/abs/2105.15203 ❌ -> Segmentation with transformer
- [x] https://www.sciencedirect.com/science/article/pii/S1746809423011400 ❌ -> segmentation
- [x] https://ieeexplore.ieee.org/abstract/document/10062649 ✅ -> ConvUNeXt, a potential better generator
- [x] https://onlinelibrary.wiley.com/doi/full/10.1002/ima.70013 ✅ -> Transformer CNN U-Net for Image to Image Generation/Translation
- [x] https://link.springer.com/article/10.1007/s11222-023-10282-8 ✅ -> Another Transformer CNN U-Net Image to Image Generation/Translation
- [x] https://arxiv.org/html/2408.06000v1 ✅ -> General Overview Image-to-Image Translation / Style Transfer


- [x] https://proceedings.mlr.press/v119/sanchez-gonzalez20a ❌ -> Have potential and also could use in this project but a bit off topic right now (more 3D Phsic Simulation with GraphNeurons)
- [x] https://www.nature.com/articles/s41598-021-01045-4 ❌ -> Sound Classification = Offtopic
- [x] https://pubs.aip.org/asa/jasa/article-abstract/157/4/3017/3344527/Sounds-of-the-deep-How-input-representation-model?redirectedFrom=fulltext ❌ -> No access (input representation in CNN Underwater Classification relevant)


- [x] https://arxiv.org/abs/1912.11474 ❌ -> Off-topic (3D Sound Propagation Roboter Search/Navigation)
- [x] https://arxiv.org/abs/2211.08570 ❌ -> Offtopic
- [x] https://arxiv.org/abs/1804.09298 ❌ -> Offtopic
- [x] https://ieeexplore.ieee.org/abstract/document/5704567 ❌ -> Off-Topic
- [x] https://arxiv.org/abs/1706.08224 ✅ -> GANs learn only partwise the true distribution (General Fact, also interesting for more complex scenarios)
- [x] https://arxiv.org/html/2403.17639v1 ❌ -> Offtopic (is about high resolution and channel transformation)
- [x] https://arxiv.org/abs/1611.07004 ✅ -> Pix2Pix GAN (basic paper)
- [x] https://arxiv.org/abs/2105.13067 ❌ -> Just High Resolution Pix2Pix
- [x] https://ieeexplore.ieee.org/abstract/document/9528943 ✅ -> General Image-to-Image Translation Overview
- [x] https://link.springer.com/article/10.1007/s12194-019-00520-y ❌ -> Just shows that generative models can handle image-to-image translation
- [x] https://cs230.stanford.edu/projects_spring_2018/reports/8289943.pdf ✅ -> WGAN is not clearly better
- [x] https://arxiv.org/abs/1704.00028 (https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490) ✅ -> Improves WGAN
- [x] https://weininghu1012.github.io/file/cpsc532L_report.pdf ❌ -> For unpaired image to image translation


- [x] https://scholar.google.de/scholar?hl=de&as_sdt=0%2C5&as_vis=1&q=graph+neural+networks+physical+simulation&btnG=
- [x] https://scholar.google.de/scholar?hl=en&as_sdt=0%2C5&q=image+to+image+translation&btnG=
- [x] https://scholar.google.de/scholar?hl=en&as_sdt=0%2C5&q=input+abstraction+deep+learning&btnG=
- [x] https://scholar.google.de/scholar?hl=en&as_sdt=0%2C5&q=input+modality+ablation+GAN&btnG=
- [x] https://scholar.google.de/scholar?hl=en&as_sdt=0%2C5&q=sound+propagation+neural+networks&btnG=

  - [x] https://link.springer.com/chapter/10.1007/978-3-030-60334-2_9 ✅ -> Image to Image in Ultrasound Images

- [x] https://scholar.google.de/scholar?hl=en&as_sdt=0%2C5&q=survey+paired+image+to+image+translation&btnG=

  - [x] https://arxiv.org/abs/1806.06397 ✅ -> Multiple U-Net Blocks as Generator for more details
  - [x] https://ieeexplore.ieee.org/abstract/document/9004254?casa_token=4yBLIPftU0AAAAAA:m7eTubb_CtqEp_ZcJG09SeL4XGEcybUVwAHF249pUahuCJ4Q290IfzCm3w2J4dFOjPxGFzHcEw ❌ (general survey image-to-image translation)

- [x] https://scholar.google.de/scholar?hl=en&as_sdt=0%2C5&q=region-focused+loss+GAN&btnG=
- [x] https://scholar.google.de/scholar?hl=en&as_sdt=0%2C5&q=physics-informed+sampling&btnG=
- [x] https://scholar.google.de/scholar?hl=en&as_sdt=0%2C5&q=importance+sampling+deep+learning&btnG=
- [x] https://scholar.google.de/scholar?hl=en&as_sdt=0%2C5&q=transformer+for+paired+image+to+image+translation&btnG=
- [x] Add Core Papers (see below)
- [x] Check references of core papers
- [x] Check the papers in https://github.com/ShenZheng2000/Awesome-Image-to-Image-Translation
  - As overview ok, but most likely for unpaired image-to-image translation and only up to 2023
- [x] Check papers: https://github.com/weihaox/awesome-image-translation/

  - [x] https://arxiv.org/abs/2302.05872 -> Diffusion for Image-to-Image Translation


=> Check the papers which are already here -> adding to zotero + summary

**AI-Guided Noise Reduction for Urban Geothermal Drilling**

```
@article{ladwig2024,
	title={AI-Guided Noise Reduction for Urban Geothermal Drilling}, 
	author={Daniel Ladwig and Martin Spitznagel and Jan Vaillant and Klaus Dorer and Janis Keuper},
	year={2024},
	url={https://opus.hs-offenburg.de/frontdoor/index/index/docId/10000},
	doi={10.60643/urai.v2024p85}
}
```
- Deep Reinforcement Learning Agent balances noise minimization and drilling efficiency by using Drill-LSTM machine state forecast and Sound-GAN sound propagation/distribution prediction
- Domain: Urban geothermal energy production -> gain/use energy from geothermal sources in high populated areas
- Problem: Drilling for installing geothermal technology generates great sound/noise pollution, while the legal requirements often cap the noise to 35dB at night while the deep drilling operations have to operate 24/7
- Sound-GAN: 3 different image-to-image generative deeplearning models where trained on 15.000 labeled data (GAN based, UNet and DDPM Diffusion)
  - Outperfomes traditional sound propagation simulations in processing speed by 50.000 times
  - Mean absolute error: 0.55dB
  - Data was normalized to: 1.0 show max dB of a component, to 0.5
  - Input: fixed grayscaled OpenStreetMap (OSM) image, where black pixels are buildings and white pixels are free space
  - Hyperparameters: 50 epochs, 20% data for testing
  - Loss:
    - GAN: combination of Binary Cross Entropy Loss & L1 Loss 
    - UNet & Diffusion: Mean Squared Error

  - Evaluation: Mean Absolute Error + Weighted Mean Absolute Percentage Error
  - GAN outperformed the other 2 models in every aspect



**Urban Sound Propagation: a Benchmark for 1-Step Generative Modeling of Complex Physical Systems**

```
@article{spitznagel2024,
	title={Urban Sound Propagation: a Benchmark for 1-Step Generative Modeling of Complex Physical Systems}, 
	author={Martin Spitznagel and Janis Keuper},
	year={2024},
	url={https://arxiv.org/abs/2403.10904},
	doi={10.48550/arXiv.2403.10904}
}
```
- 100.000 512x512 image-to-image dataset for predicting sound propagation from real 2d building maps drawn from OpenStreetmap (as a benchmark)
  - annotated with simulated sound maps (with NoiseModelling v4)
- Already have a baseline with U-Net, GAN and Diffusion -> Good in simple cases but struggle in more complex scenarios
  - Pix2Pix-base GAN
  - Standart U-Net
  - Denoising Diffusion Probabilistic
    Model
- the propagation of sound over time is described via partial differential wave equations
- Hyperparameters:
  - Each model was constructed upon a unified U-Net backbone, scaling from 64 to 1028 channels, and then
    reconverging to 64
  - Batchsize: 18
  - Epochs: 50
  - early stopping criterion (3 epochs patience) with validation loss for preventing overfitting
- Evaluation:
  - Mean Absolute Error
  - Weighted Mean Absolute Percentage Error
  - Additionally, to assess how well each model captured the reflections or diffractions we specifically measured both metrics in areas not in direct line of sight to the central signal source in the OSM images. This assessment was conducted by performing ray tracing from the sound
    source, allowing us to evaluate the models’ effectiveness in prediction propagations in Line-of-Sight (LoS) and outside Line-of-Sight (NLoS) by reflection and diffraction.
- Speed up from generative models up to 20.000 compared to simulations
- Baseline Dataset (25.000 paired datapoints)
  - Basic sound propagation without any complex propagation phenomenon
  - Geometric propagation + air absorption
  - Baseline accuracy

- Diffraction Dataset (25.000 paired datapoints)
  - Extent the base sound propagation with diffraction effects at building edges

- Reflection Dataset (25.000 paired datapoints)
  - Extent the base sound propagation with multiple reflections on building facades (counts to 3 reflection orders)

- Combined Dataset (25.000 paired datapoints)
  - Combines base sound propagation, air absorption, buidling edge diffractions and reflections from building facades
  - Valid by:
    - Temperature: -20C to +40C
    - Sound power level: 60 dB to 99 dB



**PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?**

```
@article{spitznagel2025,
	title={PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?}, 
	author={Martin Spitznagel and Jan Vaillant and Janis Keuper},
	year={2025},
	url={https://arxiv.org/abs/2503.05333},
	doi={10.48550/arXiv.2503.05333}
}
```
- the paper investigate the capabilities of modern generative models to learn detailed, complex and physically correct mappings from image pairs
- Need for new/other generative models for physical corretness (speed up is already given compared to differential equation based simulations)
- presents the *PhysicsGen* Benchmark
  - 300.000 image-pairs for 3 different taks (wave propagation, closed form lens distortion and time series prediction of motion dynamics)
- baseline evaluation with 
  - Pix2Pix
  - U-Net
  - Convolutional Autoencoder
  - Variational Autoencoder
  - Denoising Diffusion Probabilistic Models
  - Stable Diffusion
  - Denoising Diffusion Bridge Models 
- Result: High Speedup, good accuracy, fundamental problems to learn higher order physical relations
- Our analysis shows that while current generativemodels are promising significant speedups, there are still many open questions regarding their physical correctness.



**Evaluierung der physikalischen Korrektheit von Normalizing Flows bei Bild-zu-Bild Transformationen in Schallausbreitungssimulationen**

```
@article{eckerle2025,
	title={Evaluierung der physikalischen Korrektheit von Normalizing Flows bei Bild-zu-Bild Transformationen in Schallausbreitungssimulationen}, 
	author={Achim Eckerle},
	year={2025},
	url={},
	doi={}
}
```
- evaluates the physical accuracy of Normalizing Flows for image-to-image transformations in
  urban sound propagation simulations
- Used Mean Absolute Error (MAE), the Weighted Mean Absolute Percentage
  Error (wMAPE), and the Structural Similarity Index (SSIM), capturing both local deviations and
  global structural properties of the generated sound maps
- Model: "Full Glow"
- Non-Line-of-Sight (NLoS) MAE of 0.65 and Line-of-Sight (LoS) MAE of 1.84 (in baseline scenarios)
- In diffraction cases, the LoS MAE of 0.79 and the NLoS MAE of 2.63
- Multiple effects like reflection and diffraction intersect are still challenging and refining physical constraints and expanding the traindata could improve the performance
- Division of the data sets. Each of these data sets is divided into the following subsets:
  - Training dataset: 19,908 scenarios (approx. 20k, 80.0%)
  - Validation data set: 3732 scenarios (approx. 3.5k, 15.0%)
  - Test data set: 1,245 scenarios (approx. 1.5k, 5.0%)
- with 19 908 trainingsamples 108h on a NVIDIA GeForce RTX 4090 with 24 GB VRAM
- Usage of Gradient-Checkpointing not needed and slowed the training process
- Improvements:
  - More trainingdata (and with not german cities for more variety)
  - Add phsical loss to normalizing flow approach
  - Add physical details to all layers of normalizing flow




**Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks**

```
@article{isola2016,
	title={Image-to-Image Translation with Conditional Adversarial Networks}, 
	author={Phillip Isola and
              Jun{-}Yan Zhu and
              Tinghui Zhou and
              Alexei A. Efros},
	year={2016},
	url={https://arxiv.org/abs/1611.07004},
	doi={10.48550/arXiv.1611.07004},
	journal={CoRR},
    volume={abs/1611.07004},
}
```
- ...


#### Unsupervised

**CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**<br>(unsupervised...maybe not ideal)
```
@article{zhu2017,
	title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks}, 
	author={Jun{-}Yan Zhu and
              Taesung Park and
              Phillip Isola and
              Alexei A. Efros},
	year={2017},
	url={http://arxiv.org/abs/1703.10593},
	doi={10.48550/arXiv.1703.10593},
	journal={CoRR},
    volume={abs/1703.10593},
}
```
- ...



**U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization**<br>(unsupervised...maybe not ideal)

```
@article{kim2019,
	title={{U-GAT-IT:} Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation}, 
	author={Junho Kim and
              Minjae Kim and
              Hyeonwoo Kang and
              Kwanghee Lee},
	year={2019},
	url={http://arxiv.org/abs/1907.10830},
	doi={10.48550/arXiv.1907.10830},
	journal={CoRR},
    volume={abs/1907.10830},
}
```
- ...



**One-to-one Mapping for Unpaired Image-to-image Translation**

```
@article{shen2020,
	title={One-to-one Mapping for Unpaired Image-to-image Translation}, 
	author={Zengming Shen and S. Kevin Zhou and Yifan Chen and Bogdan Georgescu and Xuqi Liu and Thomas Huang},
	year={2020},
	month={March},
	url={https://openaccess.thecvf.com/content_WACV_2020/html/Shen_One-to-one_Mapping_for_Unpaired_Image-to-image_Translation_WACV_2020_paper.html},
	doi={10.48550/arXiv.1909.04110},
	booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
}
```
- ...


#### General

**An Overview of Image-to-Image Translation Using Generative Adversarial Networks**

```
@InProceedings{,
	title={An Overview of Image-to-Image Translation Using Generative Adversarial Networks}, 
	author={Xin Chen and Caiyan Jia},
	year={2021},
	month={February},
	url={https://link.springer.com/chapter/10.1007/978-3-030-68780-9_31},
	doi={10.1007/978-3-030-68780-9_31},
	booktitle={Pattern Recognition. ICPR International Workshops and Challenges},
	pages={366--380},
	isbn={978-3-030-68780-9}
}
```
- ...



**Image-to-Image Translation: Methods and Applications**

```
@article{pang2022,
	title={Image-to-Image Translation: Methods and Applications}, 
	author={Yingxue Pang and Jianxin Lin and Tao Qin and Zhibo Chen},
	year={2022},
	url={},
	doi={10.1109/TMM.2021.3109419},
	journal={IEEE Transactions on Multimedia}, 
	volume={24},
	pages={3859-3881}
}
```
- ...



**Physics-based Deep Learning**

```
@article{thuerey2021,
	title={Physics-based Deep Learning}, 
	author={Nils Thuerey and
              Philipp Holl and
              Maximilian M{\"{u}}ller and
              Patrick Schnell and
              Felix Trost and
              Kiwon Um},
	year={2021},
	month={September},
	url={https://arxiv.org/abs/2109.05237},
	doi={10.48550/arXiv.2109.05237}
}
```
- ...



**Template-Paper-Name**

```
@article{,
	title={}, 
	author={},
	year={},
	url={},
	doi={}
}
```
- template statement 1
- ...


...






---
### Data

Welche Datensätze gibt es bereits zum Thema "Schallausbreitung im Freien", "Phsyiksimulation" (auch interessant?).

**urban-sound-data**<br>
```
 -> size: 100.000 rgb images
 -> channels: r, g, b
 -> url: https://www.urban-sound-data.org/
 -> task: Urban Sound Propagation Prediction
 -> input: RGB Images? -> RED colored are the buildings
 -> label: Gray images?
 -> A benchmark (fits perfect!)
```





---

### Image-to-Image Translation Methods
for physical correctness



#### (Classical) **Generative Models**

These models learn to translate images from one domain to another by capturing the probability distribution of the target images.

- **Conditional Generative Adversarial Networks (cGANs)**
    - *Pix2Pix* → Classical cGAN for paired data
    - *Pix2Pix mit Vision Transformer*
    - *SPADE (Spatially-Adaptive Normalization GAN)* → Excels at structure-preserving translations
- **Cycle-Consistent Generative Adversarial Networks (CycleGAN & Variants)**
    - *CycleGAN* → Unsupervised translation for unpaired data
    - *CUT (Contrastive Unpaired Translation)* → Reduces overfitting and removes cycle consistency requirement
    - *UNIT (Unsupervised Image-to-Image Translation Network)* → Combines GANs with Variational Autoencoders (VAEs)
- **Variational Autoencoders (VAEs)**
  - *Conditional Variational Autoencoders (CVAE)* → Models a probability distribution over output images
- **Diffusion Models & Score-Based Generative Models**
    - *Conditional Diffusion Models* → Modern approach for controlled image transformation
    - *Score-Based Generative Models* → Uses stochastic differential equations for image synthesis
- **Normalizing Flow-Based Models**
  - *Conditional Normalizing Flows* → Uses invertible neural networks for transformations
  - *Full Glow*
- **Energy-Based Models**
    - Conditional Energy-Based Models
    - Adversarial Energy-Based Models (EBM-GANs)
    - Score-Based Conditional Generative Modeling (Diffusion als EBM)




#### **Encoder-Decoder CNNs**

These methods directly map input images to target images with a deterministic architecture.

- **U-Net & Variants**
    - *U-Net* → Widely used for segmentation and image translation
    - *Attention U-Net* → Integrates mechanisms for focusing on relevant image areas
    - *ResU-Net* → Uses ResNet blocks for deeper architecture
    - *U-Net++* → Incorporates nested skip connections for better detail preservation
  
- **Fully Convolutional Networks (FCNs)**
    - *FCN-8s, FCN-16s, FCN-32s* → Variations with different granularity
  
- **DeepLab Series**
    - *DeepLab v3+* → Combines atrous convolution with feature pyramids



#### **Convolutional Regression Networks (CRNs)**

These methods combine CNNs with regression mechanisms for precise continuous outputs.

- *ResNet + FCN* → Combines residual learning with fully convolutional networks
- *EfficientNet + Upsampling-Decoder* → Efficient architecture for high-resolution predictions
- *ViT (Vision Transformer) + Decoder* → Uses Transformer models for image-to-image regression



#### **Physic-Based Models**

These methods integrate physical principles to improve modeling.

- *PINNs (Physics-Informed Neural Networks)* → Incorporates physical equations into training
- *CNN + PDE-Solver* → Combines convolutional networks with numerical solvers for differential equations



#### **Additional Methods**

- *Hybrid Models (GAN + CNN)* → Combines advantages of both approaches
- *Graph Neural Networks (GNNs)* → Can be useful for wave propagation problems in irregular structures





> In general a higher number epochs could maybe help, many generative models need much training time



---

### Own Architecture

My idea is to take the basic Pix2Pix architecture but changing the generator part + loss.



> The loss can be advanced to WGAN-GL + PDE residual loss (e.g., Helmholtz equation: ∇2p+k2p=0∇2p+k2p=0) -> PINN-style loss. Example loss: 
> ```python
> total_loss = α * L1(pred, target)
>            + β * Helmholtz_Residual(pred, geometry)
>            + γ * Attention_Weighted_Loss(pred, target, transformer_weights)
> ```



Puzzle parts of generation part f(x) -> g(x): 

- CNN (ResNet/U-Net)
- SIREN
- FNO
- PINN
- Transformer
- FCN?



Ideas:

- <a href="#idea_siren_based">SIREN based</a>
- <a href="#idea_fno_based">FNO based</a>
- <a href="#idea_cftsm_based">CNN FNO Transformer SIREN Mix</a>
- <a href="#idea_multi_unet">Multi U-Net</a>
- <a href="#idea_multi_cftsm">Multi U-Net CNN FNO Transformer SIREN Mix</a>



> You should probably add skip-connections!



#### <a name="idea_siren_based">SIREN based</a>

Architecture:

- Input: Meshgrid coordinates (x, y), optional positional + Fourier encodings

- Network:

  - 6 × Linear(256), with sine activations

  - Final: Linear → scalar output (pressure at (x,y))

- Inference: Predict pressure for each of the 256×256 pixels

Why: Models continuous spatial functions, handles high-frequency variation, e.g., standing waves, diffraction.



#### <a name="idea_fno_based">FNO based</a>

Architecture:

- Input: 256×256×C tensor (fields, materials, geometry)

- Layers:

  - 4 × [FFT2D → Complex Weight Multiplication (K modes) → iFFT2D → Add skip + ReLU]

  - Final Layer: 1×1 Conv to map feature to output pressure

- Output: 256×256×1

Why: Efficient for global field interactions; ideal for physics with wave propagation behavior.



Example Code:

```python
from neuralop.models import FNO
import torch

model = FNO(n_modes=(16, 16), hidden_channels=64, in_channels=1, out_channels=1, dimension=2)

x = torch.randn(1, 1, 256, 256)
y = model(x)
print(y.shape)  # [1, 1, 256, 256]
```





#### <a name="idea_cftsm_based">CNN FNO Transformer SIREN Mix</a>

``````
        Input: 256x256xC (e.g., geometry, material, boundary data)
           ↓
  CNN Encoder (U-Net / ResNet-style): extracts local features
           ↓
      FNO Layer(s): models long-range frequency-aware field behavior
           ↓
  Latent Transformer Block:
                    - Attention over spatial regions
                    - Acts as a saliency filter: learns where to focus
           ↓
     SIREN Decoder (predicts continuous field from coordinates + latent features)
           ↓
        Output: 256x256x1 (e.g., sound pressure / SPL map)


Optional -> Decoder (CNN-based): upsample with skip connections
``````

Explanaiton of Components:

| Component   | Purpose                                         |
| ----------- | ----------------------------------------------- |
| CNN         | Captures **local features**                     |
| FNO         | Models **global frequency interactions**        |
| Transformer | **Learns saliency** dynamically in latent space |
| SIREN       | Refines sharp, **continuous signal** edges      |

Example Code:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# --- SIREN Layer ---
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-
                    (6 / self.linear.in_features) ** 0.5 / self.omega_0,
                    (6 / self.linear.in_features) ** 0.5 / self.omega_0
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

# --- Simple CNN Encoder ---
class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),  # 16x16
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)  # [B, C, 16, 16]

# --- Dummy FNO block (simplified) ---
class DummyFNO(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)  # Keep shape

# --- Transformer Block ---
class LatentTransformer(nn.Module):
    def __init__(self, dim, heads=4, depth=2):
        super().__init__()
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads),
            num_layers=depth
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.layers(x)
        return x  # [B, HW, C]

# --- SIREN Decoder ---
class SIRENDecoder(nn.Module):
    def __init__(self, coord_dim, latent_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(coord_dim + latent_dim, hidden, is_first=True),
            SineLayer(hidden, hidden),
            SineLayer(hidden, hidden),
            nn.Linear(hidden, 1)  # output scalar value per coordinate
        )

    def forward(self, coords, latent_features):
        # coords: [B, N, 2], latent_features: [B, N, C]
        x = torch.cat([coords, latent_features], dim=-1)
        return self.net(x)  # [B, N, 1]

# --- Full Model ---
class HybridSIRENFNOModel(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.encoder = SimpleCNNEncoder(in_channels, base_channels=64)
        self.fno = DummyFNO(512, 512)
        self.transformer = LatentTransformer(dim=512)
        self.decoder = SIRENDecoder(coord_dim=2, latent_dim=512)

    def forward(self, image, coords):
        feat_cnn = self.encoder(image)  # [B, 512, 16, 16]
        feat_fno = self.fno(feat_cnn)   # [B, 512, 16, 16]
        feat_comb = feat_cnn + feat_fno  # Combine CNN and FNO features

        # Transformer latent attention
        latent_seq = self.transformer(feat_comb)  # [B, 256, 512] if 16x16

        # Upsample latent features to match coordinates (assumes coords is Nx2 for 256x256)
        B, N, _ = coords.shape
        latent_map = latent_seq.reshape(B, 16, 16, 512).permute(0, 3, 1, 2)  # [B, C, 16, 16]
        latent_up = F.interpolate(latent_map, size=(256, 256), mode='bilinear')  # [B, C, 256, 256]
        latent_flat = rearrange(latent_up, 'b c h w -> b (h w) c')  # [B, N, C]

        # Decode
        output = self.decoder(coords, latent_flat)  # [B, N, 1]
        return output
```



#### <a name="idea_multi_unet">Multi U-Net</a>

Idea:

``````
Input Image
   │
 ┌────────────┬────────────┐
 │            │            │            
WaveEncoder   DiffEncoder  ReflEncoder
 │            │            │
WaveDecoder   DiffDecoder  ReflDecoder
 │            │            │
 └──────┬─────┴─────┬──────┘
        │   Fusion Layer (Add / Learn / Transformer)
        ▼
     Final Output
``````

Example Code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveEncoder(nn.Module):
    def __init__(self):
        super(WaveEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

class DiffEncoder(nn.Module):
    def __init__(self):
        super(DiffEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

class ReflEncoder(nn.Module):
    def __init__(self):
        super(ReflEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

class WaveDecoder(nn.Module):
    def __init__(self):
        super(WaveDecoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.upconv1(F.relu(self.conv1(x)))
        x = self.upconv2(F.relu(self.conv2(x)))
        x = self.conv3(F.relu(x))
        return x

class DiffDecoder(nn.Module):
    def __init__(self):
        super(DiffDecoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.upconv1(F.relu(self.conv1(x)))
        x = self.upconv2(F.relu(self.conv2(x)))
        x = self.conv3(F.relu(x))
        return x

class ReflDecoder(nn.Module):
    def __init__(self):
        super(ReflDecoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.upconv1(F.relu(self.conv1(x)))
        x = self.upconv2(F.relu(self.conv2(x)))
        x = self.conv3(F.relu(x))
        return x

class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer, self).__init__()

    def forward(self, wave_out, diff_out, refl_out):
        # Here, you can combine the outputs in various ways.
        # Simple average for demonstration, can be replaced with learned fusion or attention.
        return (wave_out + diff_out + refl_out) / 3

class SoundPropModel(nn.Module):
    def __init__(self):
        super(SoundPropModel, self).__init__()
        self.wave_encoder = WaveEncoder()
        self.diff_encoder = DiffEncoder()
        self.refl_encoder = ReflEncoder()
        
        self.wave_decoder = WaveDecoder()
        self.diff_decoder = DiffDecoder()
        self.refl_decoder = ReflDecoder()
        
        self.fusion_layer = FusionLayer()

    def forward(self, x):
        # Encoding paths for each effect
        wave_features = self.wave_encoder(x)
        diff_features = self.diff_encoder(x)
        refl_features = self.refl_encoder(x)
        
        # Decoding paths for each effect
        wave_out = self.wave_decoder(wave_features)
        diff_out = self.diff_decoder(diff_features)
        refl_out = self.refl_decoder(refl_features)
        
        # Fusion of all decoder outputs
        final_output = self.fusion_layer(wave_out, diff_out, refl_out)
        return final_output

# Instantiate the model
model = SoundPropModel()

# Print the model architecture
print(model)
```



#### <a name="idea_multi_cftsm">Multi U-Net CNN FNO Transformer SIREN Mix</a>

Idea:

``````
Input Image
   │
 ┌───────────────┬───────────────┐
 │               │               │            
CNN-Encoder     FNO-Encoder     FNO-Encoder
 │               │               │
SIREN-Decoder     CNN-Decoder     CNN-Decoder
 │               │               │
 └──────┬────────┴─────┬─────────┘
        │   Fusion Layer (Add / Learn / Transformer)
        ▼
     Final Output
``````



Feel free to change flexible. For example maybe SIREN, FNO and CNN (3 different). Or one path is only SIREN as direct image-to-image. 



Example Code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# SIREN Encoder for wave propagation (using Sine activation functions)
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)    # [B, 3, 256, 256] -> [B, 64, 128, 128]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # [B, 64, 128, 128] -> [B, 128, 64, 64]
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # [B, 128, 64, 64] -> [B, 256, 32, 32]
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # [B, 256, 32, 32] -> [B, 512, 16, 16]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x  # Shape: [B, 512, 16, 16]

# FNO Encoder for diffraction/reflection
class FNOEncoder(nn.Module):
    def __init__(self):
        super(FNOEncoder, self).__init__()
        # Define FNO layers or use a custom implementation for Fourier neural operator
        self.fno_layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x):
        for layer in self.fno_layers:
            x = F.relu(layer(x))
        return x

# Decoder (same structure for each effect path)
class SIRENDecoder(nn.Module):
    def __init__(self):
        super(SIRENDecoder, self).__init__()
        self.fc1 = nn.Linear(256 * 64 * 64, 4096)  # Match input size (flattened from CNN)
        self.fc2 = nn.Linear(4096, 16384)
        self.fc3 = nn.Linear(16384, 3 * 256 * 256)  # Output image size (3 channels, 256x256)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten CNN feature map
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        x = torch.sin(self.fc3(x))
        x = x.view(x.size(0), 3, 256, 256)  # Reshape back to image
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upconv1(F.relu(x))
        x = self.upconv2(F.relu(x))
        x = self.conv_out(F.relu(x))
        return x

# Fusion Layer for combining outputs of different paths
class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer, self).__init__()

    def forward(self, wave_out, diff_out, refl_out):
        # Fusion using a simple weighted sum (could be replaced by attention or learned fusion)
        return (wave_out + diff_out + refl_out) / 3

# Hybrid SIREN + FNO Model for sound propagation
class HybridSIRENFNOModel(nn.Module):
    def __init__(self):
        super(HybridSIRENFNOModel, self).__init__()
        
        # Encoders for each effect: SIREN for wave, FNO for diffraction and reflection
        self.wave_encoder = CNNEncoder()
        self.diff_encoder = FNOEncoder()
        self.refl_encoder = FNOEncoder()
        
        # Decoders for each effect
        self.wave_decoder = SIRENDecoder()
        self.diff_decoder = Decoder()
        self.refl_decoder = Decoder()
        
        # Fusion Layer
        self.fusion_layer = FusionLayer()

    def forward(self, x):
        # Encoding paths for each effect
        wave_features = self.wave_encoder(x)
        diff_features = self.diff_encoder(x)
        refl_features = self.refl_encoder(x)
        
        # Decoding paths for each effect
        wave_out = self.wave_decoder(wave_features)
        diff_out = self.diff_decoder(diff_features)
        refl_out = self.refl_decoder(refl_features)
        
        # Fusion of all decoder outputs
        final_output = self.fusion_layer(wave_out, diff_out, refl_out)
        return final_output

# Instantiate the model
model = HybridSIRENFNOModel()

# Print the model architecture
print(model)
```



Or maybe with skip connections:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN Encoder with skip connections
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)    # [B, 3, 256, 256] -> [B, 64, 128, 128]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # [B, 64, 128, 128] -> [B, 128, 64, 64]
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # [B, 128, 64, 64] -> [B, 256, 32, 32]
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # [B, 256, 32, 32] -> [B, 512, 16, 16]

    def forward(self, x):
        skip1 = F.relu(self.conv1(x))
        skip2 = F.relu(self.conv2(skip1))
        skip3 = F.relu(self.conv3(skip2))
        x = F.relu(self.conv4(skip3))
        return x, skip1, skip2, skip3  # Return encoder output and skip connections

# Decoder with skip connections
class SIRENDecoder(nn.Module):
    def __init__(self):
        super(SIRENDecoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, skip3, skip2, skip1):
        x = self.upconv1(F.relu(x))
        x = torch.cat([x, skip3], dim=1)  # Skip connection from conv3
        x = self.upconv2(F.relu(x))
        x = torch.cat([x, skip2], dim=1)  # Skip connection from conv2
        x = self.upconv3(F.relu(x))
        x = torch.cat([x, skip1], dim=1)  # Skip connection from conv1
        x = self.conv_out(F.relu(x))
        return x

# Decoder (same structure for other paths like diffraction and reflection)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upconv1(F.relu(x))
        x = self.upconv2(F.relu(x))
        x = self.conv_out(F.relu(x))
        return x

# Fusion Layer for combining outputs of different paths
class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer, self).__init__()

    def forward(self, wave_out, diff_out, refl_out):
        # Fusion using a simple weighted sum (could be replaced by attention or learned fusion)
        return (wave_out + diff_out + refl_out) / 3

# Hybrid SIREN + FNO Model for sound propagation
class HybridSIRENFNOModel(nn.Module):
    def __init__(self):
        super(HybridSIRENFNOModel, self).__init__()
        
        # Encoders for each effect: CNNEncoder for wave, FNOEncoder for diffraction and reflection
        self.wave_encoder = CNNEncoder()
        self.diff_encoder = FNOEncoder()
        self.refl_encoder = FNOEncoder()
        
        # Decoders for each effect
        self.wave_decoder = SIRENDecoder()
        self.diff_decoder = Decoder()
        self.refl_decoder = Decoder()
        
        # Fusion Layer
        self.fusion_layer = FusionLayer()

    def forward(self, x):
        # Encoding paths for each effect
        wave_features, skip1, skip2, skip3 = self.wave_encoder(x)
        diff_features = self.diff_encoder(x)
        refl_features = self.refl_encoder(x)
        
        # Decoding paths for each effect (skip connections passed)
        wave_out = self.wave_decoder(wave_features, skip3, skip2, skip1)
        diff_out = self.diff_decoder(diff_features)
        refl_out = self.refl_decoder(refl_features)
        
        # Fusion of all decoder outputs
        final_output = self.fusion_layer(wave_out, diff_out, refl_out)
        return final_output

# Instantiate the model
model = HybridSIRENFNOModel()

# Print the model architecture
print(model)
```






---

### Research

Launch title: Can generative models learn complex relationships?



Task(s): 

Improving **Physics-Based Supervised One-to-One Image Translation for Predicting Sound Propagation** with:

- new architectures
  - Vision Transformer ([Tensor-to-Image](https://github.com/YigitGunduc/tensor-to-image))
  - TransUNet (ViT + U-Net)
  - DeepONet (for Image-to-Image Translation)
- changing Feature Extraction
  - CNN (standard)
  - ResNet (pretrained(?))
  - Visiontransformer -> TransPix2Pix
- add physical loss (with PINN?) -> maybe a pretraining without physical loss and then a fine-tuning with physical loss
  - try with which models? Autenencoder? Pix2Pix? FullGlow?
- extending the dataset? (Augmentation? New Data? Other existing datasets?)
  - add augmentations
    - random cropping
    - flipping
    - rotation
    - noise injection
    - random padding
- improving Hyperparameters? (longer trainingtime needed? -> enough data for that?)
  - apply overfitting preventing techniques => Dropout, Weight Decay, Batch Normalization, Early Stopping
- using transfer learning (available?)



Based on:

- Base-Project: *AI-Guided Noise Reduction for Urban Geothermal Drilling*
- Dataset: *Urban Sound Propagation: a Benchmark for 1-Step Generative Modeling of Complex Physical Systems*
- Baseline: *PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?*
- Normalizing-Flow: *Evaluierung der physikalischen Korrektheit von Normalizing Flows bei Bild-zu-Bild Transformationen in Schallausbreitungssimulationen*



Example Physical Loss:

````python
import torch
import torch.nn as nn

class SoundPropagationLoss(nn.Module):
    def __init__(self, temperature_factor=0.002, humidity_factor=0.003, alpha=0.01):
        super().__init__()
        self.temperature_factor = temperature_factor
        self.humidity_factor = humidity_factor
        self.alpha = alpha  # Weight for regularization

    def forward(self, predicted_image, source_power, temperature, humidity, distance_map):
        """
        Computes a physics-informed loss based on sound propagation principles.
        
        Arguments:
        - predicted_image: Tensor of predicted sound intensity (grayscale image).
        - source_power: Scalar, initial power level in dB.
        - temperature: Scalar, ambient temperature.
        - humidity: Scalar, relative humidity.
        - distance_map: Tensor of distances (same shape as predicted_image).
        
        Returns:
        - physics_loss: Scalar loss value.
        """

        # Compute geometric divergence loss (logarithmic distance loss)
        div_loss = 20 * torch.log10(distance_map + 1e-6)  # Add small value to prevent log(0)

        # Compute atmospheric absorption loss (depends on temp & humidity)
        alpha = self.temperature_factor * (temperature / 300) + self.humidity_factor * (humidity / 50)
        atm_loss = alpha * distance_map

        # Compute expected received intensity
        expected_image = source_power - div_loss - atm_loss

        # Compute loss (MSE between expected and predicted)
        physics_loss = nn.MSELoss()(predicted_image, expected_image)

        return self.alpha * physics_loss
````

And another example:

```` python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace  # For PDE approximation

# Decoder/ Generation -> only an example
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        
        self.encoder = nn.Sequential(
            conv_block(in_channels, 64),
            nn.MaxPool2d(2),
            conv_block(64, 128),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            conv_block(64, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
            nn.Sigmoid()  # Normalize to [0,1]
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# === loss ===
def laplacian_2d(u):
    """Computes the discrete Laplacian using finite differences."""
    return torch.tensor(laplace(u.numpy(), mode='reflect'))

def wave_equation_residual(u, c=343.0, dt=0.01):
    """Computes the residual of the wave equation: ∂²u/∂t² - c²∇²u = 0"""
    laplacian = laplacian_2d(u)
    residual = (u - 2 * u + u) / dt**2 - c**2 * laplacian
    return residual

def compute_loss(pred, target, lambda_pde=0.1):
    mse_loss = nn.MSELoss()(pred, target)
    
    # Compute PDE residual loss
    pde_residual = wave_equation_residual(pred)
    physics_loss = torch.mean(pde_residual**2)
    
    total_loss = mse_loss + lambda_pde * physics_loss
    return total_loss

# === Traning ===
# Initialize model, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example Training Loop
for epoch in range(100):
    optimizer.zero_grad()
    
    # Example input: Urban sound map
    urban_map = torch.rand(1, 1, 128, 128).to(device)  # Random placeholder data
    target_wavefield = torch.rand(1, 1, 128, 128).to(device)  # Ground truth
    
    pred_wavefield = model(urban_map)
    loss = compute_loss(pred_wavefield, target_wavefield)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
````

DeepONet:

````python
import torch
import torch.nn as nn

class DeepONet(nn.Module):
    def __init__(self, img_size=64, in_channels=3, branch_hidden=128, trunk_hidden=128):
        super().__init__()
        # Branch network (learns the function input)
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(branch_hidden, branch_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Trunk network (learns the output coordinate space)
        self.trunk = nn.Sequential(
            nn.Linear(2, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, branch_hidden)
        )

        # Final projection
        self.fc = nn.Linear(branch_hidden, 3)  # Output RGB image values

    def forward(self, x, coords):
        branch_out = self.branch(x)  # Extract global features
        trunk_out = self.trunk(coords)  # Learn spatial dependencies
        combined = branch_out * trunk_out  # Merge features
        output = self.fc(combined)  # Map to image space
        return output.view(x.shape[0], 3, x.shape[2], x.shape[3])  # Reshape to image

# Example usage
model = DeepONet()
input_image = torch.randn(4, 3, 64, 64)  # (batch, channels, height, width)
coords = torch.randn(4, 2)  # Spatial coordinates
output_image = model(input_image, coords)
print(output_image.shape)  # Expected: (4, 3, 64, 64)
````

Vision Transformer:

````python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, hidden_dim=256, num_layers=6, num_heads=8):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)
        self.out_conv = nn.ConvTranspose2d(hidden_dim, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        patches = self.patch_embed(x)  # In Patches aufteilen
        patches = patches.flatten(2).transpose(1, 2)  # (batch, num_patches, hidden_dim)
        patches += self.pos_embed  # Positionale Einbettung

        encoded = self.encoder(patches)  # Transformer-Encoder

        decoded = self.decoder(encoded)  # In Pixel-Werte zurückprojizieren
        decoded = decoded.transpose(1, 2).view(batch_size, -1, height // 8, width // 8)
        output = self.out_conv(decoded)  # In Bild umwandeln
        return output

# Beispielnutzung
model = VisionTransformer()
input_image = torch.randn(4, 3, 64, 64)
output_image = model(input_image)
print(output_image.shape)  # Erwartet: (4, 3, 64, 64)
````

TransUNet:

````python
import torch
import torch.nn as nn

class TransUNet(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, hidden_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.upconv1 = nn.ConvTranspose2d(hidden_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        patches = self.patch_embed(x)  # Bild in Patches umwandeln
        patches = patches.flatten(2).transpose(1, 2)  # (batch, num_patches, hidden_dim)
        patches += self.pos_embed  # Positionale Einbettung

        encoded = self.encoder(patches)  # Transformer-Encoder

        # Reshape zurück ins Bildformat
        h, w = height // 8, width // 8
        encoded = encoded.transpose(1, 2).view(batch_size, -1, h, w)

        # U-Net Decoder
        x = self.upconv1(encoded)
        x = self.upconv2(x)
        output = self.out_conv(x)
        return output

# Beispielnutzung
model = TransUNet()
input_image = torch.randn(4, 3, 64, 64)
output_image = model(input_image)
print(output_image.shape)  # Erwartet: (4, 3, 64, 64)
````

Custom ResNet Transformer Model:

````python
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, resnet_model):
        super(ResNetEncoder, self).__init__()
        self.encoder = nn.Sequential(*list(resnet_model.children())[:-2])  # Removing the last layers
        
    def forward(self, x):
        return self.encoder(x)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, hidden_dim):
        super(TransformerDecoder, self).__init__()
        self.transformer = nn.Transformer(d_model=embed_size, num_heads=num_heads, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, hidden_dim)
    
    def forward(self, x, memory):
        x = self.transformer(x, memory)
        return self.fc_out(x)

class ImageTranslationModel(nn.Module):
    def __init__(self, resnet_model):
        super(ImageTranslationModel, self).__init__()
        self.encoder = ResNetEncoder(resnet_model)
        self.decoder = TransformerDecoder(embed_size=256, num_heads=8, num_layers=6, hidden_dim=64)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)  # using the same encoded features as memory
        return decoded
````



---

### Commands

Running the simulation software:

``````
docker build -t noisemodellingapi:test .
``````

``````
docker run -v "$(pwd)/output:/app/debug_output" noisemodellingapi:test --buildings "[[[0,0], [0,50], [50,50],[0,0]], [[0,0], [0,80], [40, 80], [40,0], [0,0]]]" --linux --n_samples 1
``````

-> see *create_custom_dataset_multiprocessing.py* for all options



Create a *clear.sh*:
```bash
# find and kill all multiprocessing.py processes
echo "Find and kill all multiprocessing.py"
ps aux | grep multiprocessing.py | grep -v grep | awk '{print $2}' | xargs kill -9

echo "Find and kill all running docker container"
# kill all running docker container
docker kill $(docker ps -q)

# clear docker
echo "Clean Docker"

# Prune everything
docker container prune -f
docker image prune -a -f
docker volume prune -f
docker network prune -f
docker builder prune --all -f
```

Create a *run.sh 64*:
```bash
# Initialize Conda
eval "$(conda shell.bash hook)"

# if you don't have the environment run this:
# conda create -n "$ENV_NAME" -y
# conda activate "$ENV_NAME"
# conda install python=3.9 -y
# conda install pip -y
# pip install -r ./requirements.txt

# Activate the environment
conda activate "$ENV_NAME"

mkdir -p ./logs

# Check if a worker argument was provided
if [ -n "$1" ]; then
    WORKER_ARG="--worker $1"
else
    WORKER_ARG=""
fi

# Run the simulation  multiprocessor
nohup python multiprocessing.py $WORKER_ARG > ./logs/output.log 2>&1 &
```




---
### Progress

**März 2025:**
- read 4 core papers
- thought about possible research questions and optimizings
- started with the noisemodelling simulation software
- setted 2 first experiments
  - Train 2x models, one which gets simple sound propagation as input and the other gets the plain input -> which perform better on the reflexion (sub)dataset
  - Adaptive sampling -> train data should contain much more complex examples + add regularizations against overfitting

- created simple sound propagation dataset with equal building setup for basic and reflection simulation (comming)
- parsing arguments all with one yaml file (without python arguments) and with pydantic (validation)
- parsing all arguments over arguments to the scripts and do not use config file in scripts
- fixing horizontal diffraction -> it is a flag argument, set or not set
- allowing multiple settings: -> new naming ```id = {i}_{cur_reflection_order}_{cur_diffraction_order}_{cur_max_reflection_dist}_{cur_max_src_dist}```
  - horizontal diffraction
  - reflection order
  - max reflection distance
  - max src distance

- added 2 running scripts for easy running (one for unbounded running, so that the local system can be exited)
- added main.py which is used as a central startpoint
  - loads config
  - running the underlying script (pipeline.py or create_custom_dataset_multiprocessing.py)

- new feature summary: multiple settings, united arguments parsing, argument validation, central start script
- adjusted pipeline.py/Controller and create_custom_dataset_multiprocessing.py to work with the new config/parsing + work with multiple settings
- config is passed as an object: config does not have to be created via yaml, can also be created in python or via python arguments, should be implemented quickly and work easily
- most changes in: ConfigHandler, create_custom_dataset_multiprocessing.py, pipeline.py, Controller.py
- create_custom_dataset_multiprocessing.py tested and do work! + pipeline.py is not tested due to missing locations_ca.csv

**April 2025:**

- started data generation of 10.000 datapoints and 4 different datasets with varying diffraction/reflection -> 1, 2, 3, 4, 5 buildings
- "random" error from time to time...there come errors with the database -> trying to delete every id files in the run for a good/working retry
- added multiprocessing config creation method
- maintain datageneration
- wrote script for extracting single datasets from multiple once
- prepared "planning" presentation
- trained 4x Pix2Pix on the 4 datasets
  - parameter: 
    - lr = $1\dot10^(-4)$
    - lr_policy = cosine (to avoid local minima in higher epochs)
    - optimizer: Adam
    - adam beta = 50%
    - epochs = 250 (upgrade from 50 -> to at the same time see, if a higher epoch amount can help further)
    - GAN Mode: lsga (uses Mean Squared Error not Cross Entropy)

- ...

**Mai 2025:**

- ...

**Juni 2025:**
- ...

**Juli 2025:**
- ...

**August 2025:**
- ...

**September 2025:**
- ...

**Oktober 2025:**
- ...

**November 2025:**
- ...

**Dezember 2025:**
- ...

**Januar 2026:**
- ...

**Februar 2026:**
- ...

**März 2026:**
- ...

**April 2026:**
- ...

**Mai 2026:**
- ...

**Juni 2026:**
- ...






---
### Meetings



**18.07.2025 Weekly Meeting**<br>

Participants: Keuper, Martin, Ich<br>
Location: Zoom<br>
Time: 13:00 O'Clock

- ...



**18.07.2025 Weekly Meeting**<br>

Participants: Keuper, Martin, Ich<br>
Location: Zoom<br>
Time: 13:00 O'Clock

- cosmos auf server ausprobieren -> server linux -> fine-tuning
- Bei pius nächste woche abstimmen -> hiwi



**27.06.2025 Weekly Meeting**<br>

Participants: Keuper, Martin, Ich<br>
Location: Zoom<br>
Time: 13:00 O'Clock

- (Physical loss -> law of conservation of energy, ... -> ? Possibly applyable)
- Check losses again in more detail -> from good results (only reflection) -> why such a big jump
  - Which losses/weighting is the reason?
  - Send result/weighting
- Read paper for Accukstik propagation in more detail + send
  - https://acoustique.ec-lyon.fr/publi/alguacil_aiaaj22.pdf
  - https://www.sciencedirect.com/science/article/pii/S0022460X21003527
- https://github.com/ArshKA/PhysiX
  - Try out + read in
- https://github.com/nvidia-cosmos/cosmos-predict1

Translated with DeepL.com (free version)



**06.06.2025 Weekly Meeting**<br>

Participants: Keuper, Martin, Ich<br>
Location: Zoom<br>
Time: 13:00 O'Clock

- Showed Depth Anything Results
  - Seems like too much variance to learn
- Sample Masking (Random? Only weighted? Maybe after x epochs?) -> Just thoughts about which we talked
  - Using the non line of sight as mask -> Look how the eval.py calcs the non line of sight error and use nlos as mask
  - Maybe using the Complex - Base as Mask
- The idea is to reduce the complexity of the task until it is learnabe
  - Which variance is learnable
  - Maybe even splitting into only angle values
  - Only Predicting Reflection (seperated) > only predicting reflections masked on the parts where reflections are > Only on a few simple buildings > ...
- Task for the week
  - With few Buildings dataset:
    - Predict only seperated reflection
    - Predict only seperated reflection with mask onto reflections (threshold at 0.2 or something)
- Check out Pix2Pix Residual result
- Write/send results directly (Martin + Keuper on conference in nashville)
- Fusion Head only 1x1 kernel?



**30.05.2025 Weekly Meeting**<br>

Participants: Keuper, Ich<br>
Location: Zoom<br>
Time: 13:00 O'Clock

- Quantization Error found: cv2 imreads as singed int and minus values overflows -> so convert type to unsigned float
- New Modelarchitecture tried out (implemented data for it)
  - One Model (DepthAnything/Pix2Pix), input: osm, target: baseline propagation
  - One Model (DepthAnything/Pix2Pix), input: osm, target: reflection - baseline propagation (= only reflections)
  - One fusion model to fusion the two outputs from the models
  - Added Loss for small errors
  - Changed output of depth anything to sigmoid (0 - 1)
  - And other small adjustments...

- Test Pix2Pix GAN + Depth Anything with ComplexOnly / Split Tasks architecture 



**23.05.2025 Weekly Meeting**<br>

Participants: Keuper, Martin, Ich<br>
Location: Zoom<br>
Time: 13:00 O'Clock

- Fixing plotting -> other cmap with log-scale (and scale back to 255) OPTIONAL

  - Try Log (before and after invert -> not at the same time!):

    ```python
    img = np.log1p(img.astype(np.float32))
    img = img / img.max() * 255
    img = img.astype(np.uint8)
    ```

     

  - Try out gamma correction:

    ````python
    gamma = 0.4  # <1 enhances shadows, >1 compresses shadows
    img = img.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    img = (img * 255).astype(np.uint8)
    ````

  - Histogramm equalization

    ````python
    img = cv2.equalizeHist(img)
    ````

    or:

    ````python
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    ````

- Solo eval -> split MAE and MAPE visualization

- Add log scaled MAE to the losses? -> or other loss for more detail

  ````python
  class LogL1Loss(nn.Module):
      def __init__(self, eps=1e-8):
          super().__init__()
          self.eps = eps  # Prevent log(0)
  
      def forward(self, y_pred, y_true):
          log_pred = torch.log(y_pred + self.eps)
          log_true = torch.log(y_true + self.eps)
          return torch.mean(torch.abs(log_pred - log_true))
  ````

  ````python
  class MAPELoss(nn.Module):
      def __init__(self, eps=1e-8):
          super().__init__()
          self.eps = eps  # Avoid division by zero
  
      def forward(self, y_pred, y_true):
          return torch.mean(torch.abs((y_true - y_pred) / (y_true + self.eps))) * 100
  ````

  

- Finding and Fixing Quantization Error:

  - 1d profile -> error in location -> input + => with artifacts + without artifacts
    - View a row/column from an image array -> identy

  - Display a concrete chain -> input, prediction, ground truth, difference
  - Test with base simulation (wgan-gp) -> such errors must not be in there
  - Metric correct, but visualization wrong? -> any errors due to inversion?
    - Possibly clarified if CMAP works

  - Only at created dataset or also in physgen dataset?

- After fixing the bug, we can think about how to do/apply the mask/sample maps



**16.05.2025 Weekly Meeting**<br>

Participants: Keuper, Martin, Ich<br>
Location: Zoom<br>
Time: 13:00 O'Clock

- Implementing Depth Anything and try out training + evaluation (self assigned)
  - Maybe try to implement into the forked pix2pix repo -> https://github.com/xXAI-botXx/paired_image-to-image_translation/ from https://github.com/DepthAnything/Depth-Anything-V2
  - Without Conditioning -> Sound Source is fix
  - The Pipeline of them
  - The Pipeline of Pix2Pix

- Make a test training with the parameters from the Physgen paper (self assigned)
  - Batchsize: 18
  - Learnrate: 1e-4 (Generator: 2e-4)
  - Epochs: 50
  - Lambda L1: 100
  - Lambda GP: 10
  - Optimizer: Adam
  - Adam betas:  (0.5, 0.999)
  - Results to reach:
    - LoS MAE = 0.91
    - NLoS MAE = 3.36
    - LoS wMAPE = 3.51
    - NLoS wMAPE = 18.06 

- Base Simulation vs Reflection Simulation -> warum quantisierter? Search/Fix Quantisier Fehler
  - Auch ring artefakte damit vlt erklärebar
  - Evtl exportier speicher fehler
  - Bildformat -> histogramm anschauen
  - Mit Physgen evaluieren

- WGAN-GP + Input Image Training Evaluation 



**08.05.2025 Weekly Meeting**<br>

Participants: Keuper, Martin, Ich<br>
Location: C108<br>
Time: 13:00 O'Clock

- Improve plots:
  - Add Ground Truth to the Result / Example Plots
  - Log Scale the plots (to see the details better)
  - Plot the difference of ground truth and the prediction (so you get the problem zones)

- Test Line of Sight / Out of Sight Error -> With Base Simulation as Input a very small line of sight error expected
  - Need OSM image (Satalite image) as additional input

- Test which error the base simulation as input has -> does the model even improven it?
- Try Wasserstein GAN GP
  - Implement WGAN-GP as optional parameter in Pix2pix GAN
  - One Training with and one without
  - Check the result + was the training more balanced?




**02.05.2025 Weekly Meeting**<br>

Participants: Keuper, Ich<br>
Location: Zoom<br>
Time: 13:00 O'Clock

- Focus first on the "Complex Focus Only"-Experiment and not on the paper research, because the research is only needed if this approach does have any successfull results

- Next Friday results of the experiments

  - Create a test dataset (1000 Images)
  - Train all Modelcombinations with Hyperparameter from PhysGen Paper (BatchSize 18, Epochs 50, Learnrate 0.001)
  - Create all predictions on test dataset
  - Evaluate (use eval script fro PhysGen)

  

**25.04.2025 Weekly Meeting**<br>

Participants: Martin, Ich<br>
Location: Zoom<br>
Time: 13:00 O'Clock

- Use "Weights and Bias" ->[website](https://wandb.ai/site/) , register + give API code over parameter by call => maybe use
  - for simple experiment tracking and result analysis/comparison

- Make training parameter tests (only on one dataset and later the best option apply for all) -> removing artifacts
  - Validation Dataset create and use (maybe 500 images -> create with your pipeline)
  - Hyperparameter tuning
    - reduce amount of epochs (50, 100)
    - increase amount of epochs (500)
    - change learnrate schedular from cosine to linear
    - change batchsize to 1
    - try batchsize as in the PhysGen paper
  - Adding Wasserstein Loss?
  - Check results at the log file, "Weights and Bias" or/and on test data 
- Currently conzentrate on literature research (search + summary + importance/relevance -> have to be checked)
- Martin will check/use the standard pipeline in the new adjusted pipeline -> he will probably merge if everything works + writes you if something is wrong/need help
- Question to Janis Keuper: What should be included in the literature search / document? Found papers, their take-aways and why they could be/are relevant?



**22.04.2025 Planning Presentation**<br>

Participants: MAR 1 AI<br>
Location: Zoom<br>
Time: 14:00 O'Clock

- in future could try to use normalizing flows
- Experiment 2 correction:
	- based on experiment 1
	- search physical complex areas + mask all other areas, so that the loss is concentrated on the complex areas
- Liteartur Tool -> Research Paper/Literature => Bibtex + Latex -> finish until: 13 Mai
- Data Science Project will be clarified (propably something with LLM)



**03.04.2025 Weekly Meeting**<br>

Participants: Keuper, Martin, Ich<br>
Location: C108<br>
Time: 13:30 O'Clock

- Primary Key Missing Error: Error can be another, often consequential error. Groovy just keeps going, even though nothing comes out right
- Pull error loop even one higher = reload everything (reset start_index in config)
    - is propably not useful, there is nothing happening over that
- Work out a plan for the semester 
    - Familiarization with framework (adaptation)
    Carry out experiment 1
    - Literature for experiment 2 (sampling)
    - Implement literature or own sampling strategies
- Title for 2. experiment/paper: Physical Informed Sampling
    - Samples categorization in "how physical as difficult" -> either calculate or estimate (show that estimation already works very well)

- Generate extra heavy data (possibly later)
- New meeting slot: Friday 1 pm



**27.03.2025 MAR KI Meeting**<br>

Participants: Keuper, Dorer, Sven, Louis, Paul, Lukas, Ich<br>
Location: C108<br>
Time: 13:30 O'Clock

- Tomorrow presentation of Masters in Campus West (8 Uhr)
- Data Science Class can be visited but don't have to be visited
- 50% project => report (progress report, will be discussed in more detail...literature work) + probably 20% pitch (sell topic)
- 3 dates + today's
  - Make slides (presentation) for the meetings
  - upload slides -> a few slides are enough
- 1st date: Work packages this semester + how to proceed (informal presentation) -> 22.04.2025 2 pm
- 2nd literature presentation (informal)
- 3rd final report (how did it go? ...) -> graded
- grading for everything as a whole (rather exchange, no grade stress)
- master thesis, focus can shift again
- goal for the master: submit publication (make paper) or more ...
  - for example I could publicate a paper about my results of the image to image prediction in geotermal drilling context and then continue with another (maybe similiar) topic
- My master can also maybe go about other projects -> first geotermal drilling, but then maybe it will change
- Everbody told about his project. Me:
  - AI drill project
  - Drilling in inner cities -> Geothermal energies
  - AI controlled -> quieter
  - Subtask: Prediction of sound propagation
  - Simulation too slow
  - Generative models faster but less precise
  - How to optimize/improve generative models
- Sven: Magma Reinforcment Learning of 3 Robot types with only one model and a one-hot-encoded number => learning together from each other (?)
- Paul: Uniklinik Freiburg, "Magic Pen" like semantic segmentation with U-net and persons first input (one organ have many picture slides)
- Lukas: Sick explanaible decision making during classification, to get closer to varyfying machine learning apporaches in critical robotic situations
- Louis: Baden-IT -> Support Agent System via LLM and with context learning of solutions for problems



**27.03.2025 General Progression**<br>

Participants: Keuper, Martin, Ich<br>
Location: C108<br>
Time: 13:30 O'Clock

- next steps:
  - adjust arguments, so that they are consistent: use pydantic and give bash configs over python arguments
    - Also common pythom arguments over pydantic? -> yes (not 100% sure, ask!)
    - fix that all arguments come with the same way
  - adjust random shape generation pipeline to run simulation with same shapes and different settings (reflexion, ...)
  - generate 10.000 datapoints -> 10.000 with basic physic + 10.000 with reflexion 1
  - make experiment with [pix2pix GAN](https://github.com/LouisW2202/pytorch-CycleGAN-and-pix2pix) -> experiment checks if noisemap without physics improve performance over normal image as input
- in future should have a own master repo (this here? or from the university?)



**21.03.2025 Software Tool Einführung:**<br>
Participants: Martin, Ich<br>
Location: Zoom<br>
Time: 10:00 O'Clock

- the noise simulation works with receivers. on the location of a receiver the sound volume will get calculated. The *delta*-distance says how many receiver should there every x steps (receiver grid) + there come additional receiver on every buildings edges

- the noise simulation software got a bit expanded to allow receiver grid and receiver on the edes of buildings

- a simulation process creates a local database for the data and so only one process at the time is possible. To still be bale to make multiple generation processes the project uses multiple docker containers to be able to run multiple simulations at the time + they got used with computing devices in a kubernetes cluster but now only one note of this cluster is in use with about 250 kernels for computing

  - connect to the note/cluster/server via SSH and VS Code

- the normal simulation process is: GPS data + optionally temperatur, humidity, ... = satelite image + sound propagation image 

- the simplified simulation process is: shapes = satelite image + sound propagation image
  - Only a few shapes are spawned randomly to keep it more simple (example usecase: testing)
  - create_custom_dataset....py
  - Available buildings can be processed by the run command or with a json file -> both have to processed in the --buildings parameter 

- there are 4 cases: basic, reflection, diffraction, combined
  - basic: simple wave propagation
  - reflection: basic with bouncing from buildings (order decides how often it bounces) -> similar to ray-tracing
  - diffraction: how the sound waves are moving behind a building (around it)
  - combined: basic + reflection + diffraction + other properties (temperature, humidity, ...) -> will be added as additional channel for each property

- run command: 
  
  ``````
  docker build --no-cache -t noisemodellingapi:test .
  ``````
  
  ```cmd
  docker run noisemodellingapi:test --buildings "[[[0,0], [0,50], [50,50],[0,0]], [[0,0], [0,80], [40, 80], [40,0], [0,0]]]" --linux
  ```
  
- I have to create: the same constellation with basic and reflection -> maybe I can create both in the same simulation run, or I create the exact building location and create the 2 simulations with it, or I run it as basic and then extract and use the exact building spwan and give it to the next simulation with reflection on -> or think about it...there are many possible solutions 

  - but the simulation/wrapper/process must be adjusted so fixed locations are the input
  - look how the students made that the input is not GPS else the shape

- config.cfg contains standard parameters (delta distance, reflection, ...)

- 256x256 size is standard (I can also only use still)

- standard: lwd500 =  95dB (-> 500 hz)

- LAEQ = is the right output format, ignore LEQ (adjusts the frequences to the human hearing) -> you can turn this off

- look at the project, the scripts and how it works (look at both branches, the GPS and the shape way)

  - look at the changes

- scripts folder: wrapper for controlling the simulation -> if you pass --linux the bash files will be running, else the bat files

- to do:

  - try to sample some datapoints with the simpler generation (1 or 2 samples) via local docker
  - can the python wrapper be changed to directly control the simulation and not the bash scripts?
  - could the 2 branches get merged? it would be nice to control the input as parameter
  - CONF_DIFF_HORIZONTAL -> pass no value = false and pass any value =  true BUT should be 0=false and 1=true  (diffraction = sound propagation moving around buildings)

- Martin do:
  - add me to github
  - send noise modelling simulation software (with small adjustment to the grid creation)
  - in future the convert exe might be needed
  
- good start/example: example_usage.ipynb

- simulation controllable over python wrapper

  - python wrapper( batch-files( Gruvy-Files ) )

- program to show shape files: OGIS

- idea could be RGB images as input so that the AI can learn to use more environment features

  - problem: the labled data does not contain such environment data (trees, ground information, ...)
  - with real world sound propagation data possible, else not -> or a more advanced sound propagation synth dataset with unreal if such is possible

- source definition in source_config.geojson

- when loading the input the projection type of the given coordinates are passed, most likely using the EPSG:3857 projection = it will interpret the coordinates as GPS format

- Pix2Pix GAN is good for the experiments (a image loader is already available in [this repo](https://github.com/LouisW2202/pytorch-CycleGAN-and-pix2pix)



**20.03.2025 Zweites Meeting:**<br>
Participants: Keuper, Martin, Ich<br>
Location: C108<br>
Time: 13:30 O'Clock

- There is a less complex dataset + describing document for it
- Introduction to the simulation/dataset creation is important -> stated tomorrow
- 2 first Approaches
  - Train 2x models, one which gets simple sound propagation as input and the other gets the plain input -> which perform better on the reflexion (sub)dataset
  - Adaptive sampling -> train data should contain much more complex examples + add regularizations against overfitting
- Start with the stepped input experiment on simpler dataset (maybe have to generate new data)



**10.03.2025 Startgespräch Herr Keuper und ich:**<br>
Participants: Keuper, Ich<br>
Location: Zoom<br>
Time: 10:00 O'Clock

- Input/Output: Bild zu Bild Generierung
- Paper lesen (4 Stück)
- 1 Mal pro Woche zusammensetzen
- Bei Meetings mit dabei (Dienstags und Donnerstags) -> intern, Doktoranten und bei Herrenknecht
- Fächer: 
  - Studium Generale (Mi. 9:45-11:15 Uhr A303)
  - Akt Meth Masch Lernen (Do. 14:00-17:15 Uhr A301) -> Aktuelle Methoden des Maschinellen Lernens und deren Anwendung
  - Einmalig: Erstsemestereinführung (Di. 18.03 9:30-10:30 Uhr E311/E412/E008)
  - Nicht Data Science! -> Herr Keuper kümmert sich um die Anrechnung

- Bei Fragen gerne im Google Chat schreiben
- Unklarheiten: 
  - Aufgabenstellung/Thema: Was soll nun genau im Master getan werden? -> Weitere Modelle und Experimente unternehmen






---
### Fragen

- Aufgabenstellung/Thema: Was soll nun genau im Master getan werden? -> Weitere Modelle und Experimente unternehmen
-  What if we would give an RGB image as input -> the model could maybe learn to use more than only binary (building/not building) -> other objects also behave with the sound propagation 
-  4 Models for 4 tasks or 1 model for 4 tasks? -> I think 4 models for 4 tasks






---
### Das sagt die KI

````
Below is an overview of the main classes of generative models—and some representative methods—you could consider for image-to-image translation in your task (converting urban satellite RGB images into grey-scale maps representing sound propagation):

---

### 1. **Conditional Generative Adversarial Networks (cGANs)**
- **Pix2Pix:**  
  Designed for paired image translation tasks, Pix2Pix uses a conditional GAN framework where the generator learns to map an input image to an output image (with the discriminator helping enforce realism). This method is particularly well-suited if you have aligned pairs of satellite images and corresponding sound propagation maps.  
  *Key Reference:* Isola et al., “Image-to-Image Translation with Conditional Adversarial Networks.”

- **Other cGAN Variants:**  
  Many modifications and improvements on the original Pix2Pix have emerged, such as using different architectures (e.g., U-Net-based generators) or loss functions (e.g., adding perceptual or L1/L2 losses) to better capture fine details or regression nuances.

---

### 2. **Unpaired Image Translation Methods**
If paired data (i.e., perfectly aligned input-output pairs) is scarce, you might consider models that work with unpaired datasets:

- **CycleGAN:**  
  Uses cycle-consistency losses to learn mappings between two domains even when paired examples are not available. This can be useful if you can collect separate sets of satellite images and sound propagation maps.
  
- **UNIT (Unsupervised Image-to-Image Translation):**  
  Assumes a shared latent space between the two domains and learns a mapping through this common representation.
  
- **MUNIT (Multimodal Unsupervised Image-to-Image Translation):**  
  Extends UNIT to allow for multimodal outputs, which might be useful if your sound propagation mapping has inherent ambiguities.
  
- **CUT (Contrastive Unpaired Translation):**  
  A more recent method that uses contrastive learning to preserve the content of the input image while changing its style according to the target domain.

---

### 3. **Autoencoder-Based Approaches**
- **Conditional Variational Autoencoders (CVAE):**  
  CVAEs incorporate the conditioning signal (your satellite image) into the encoding-decoding process. They tend to produce smoother outputs and can be easier to train than GANs, although they might lack some of the sharpness achievable with adversarial training.

- **U-Net Architectures:**  
  While not generative models by definition, U-Nets (often used in segmentation tasks) are widely incorporated in the generator part of GANs (e.g., Pix2Pix) because their skip connections help retain high-resolution details from the input.

---

### 4. **Diffusion Models and Score-Based Generative Models**
- **Conditional Diffusion Models:**  
  Recently popularized for their ability to generate high-quality images, these models gradually denoise a noisy image in a process conditioned on the input. They have been applied successfully to image-to-image tasks and might be adapted for your regression-like output.
  
- **Score-Based Models:**  
  These models learn to generate images by estimating the gradient (score) of the data distribution. Like diffusion models, they can be conditioned on an input image to guide the generation process.

---

### 5. **Normalizing Flows**
- **Conditional Normalizing Flows:**  
  These are invertible models that learn the exact likelihood of the data. When conditioned on an input image, they can be used for image-to-image translation tasks and offer exact likelihood estimation, which might be beneficial for understanding uncertainty in sound propagation outputs.

---

### Considerations for Your Task
- **Data Availability:**  
  - **Paired Data:** If you have a dataset of satellite images and corresponding sound propagation maps, methods like Pix2Pix or CVAEs are natural choices.  
  - **Unpaired Data:** If the datasets are collected separately, CycleGAN, UNIT, or CUT might be more applicable.
  
- **Output Nature:**  
  Since your output is a grey-scale map representing sound propagation (likely a continuous-valued field), consider modifying loss functions (e.g., using L1/L2 losses) or network architectures to better handle regression tasks rather than categorical outputs.

- **Physical Consistency:**  
  Depending on the complexity of the sound propagation phenomena, you might want to integrate domain-specific knowledge (e.g., physics-based constraints) into the training process, either by designing custom loss functions or hybrid models.

- **Recent Trends:**  
  Diffusion and score-based models have recently shown impressive results in generating detailed and high-fidelity images. They might be particularly attractive if you are looking for state-of-the-art quality and are comfortable with the typically higher computational cost of training these models.

---

### Summary
In summary, your options include:

- **Paired Approaches:** Pix2Pix and its cGAN variants, Conditional VAEs, and U-Net based architectures.
- **Unpaired Approaches:** CycleGAN, UNIT, MUNIT, and CUT.
- **Modern Alternatives:** Conditional diffusion models, score-based generative models, and conditional normalizing flows.

The best choice depends on your specific dataset (paired vs. unpaired), the nature of the mapping from satellite imagery to sound propagation, and the desired trade-offs between training complexity and output fidelity.
````



```
Um die Schallausbreitung einer Bohrung in einer Stadt zu berechnen und herauszufinden, wo es wie laut ist, gibt es verschiedene mathematische Ansätze, abhängig von der Genauigkeit und dem Maßstab. Hier ist eine schrittweise Vorgehensweise mit verschiedenen Methoden:


---

1. Modellierung der Schallquelle

Die Bohrung kann als punktförmige oder flächenförmige Quelle modelliert werden. Für eine Punktquelle gilt die Kugelausbreitung:

L_p(r) = L_w - 20 \log r - 11

wobei:

 der Schalldruckpegel in dB bei Abstand ,

 die Schallleistung der Quelle in dB,

 der Abstand zur Quelle in Metern,

 die Konstante für die Umrechnung von Schallleistung auf Schalldruck in Luft.


Wenn es sich um eine Linienquelle (z. B. entlang einer Straße) handelt, fällt der Pegel nur mit 10 \log r statt 20 .


---

2. Berücksichtigung von Dämpfung

a) Luftabsorption

Schall wird durch die Luft gedämpft, abhängig von der Frequenz:

L_p (r) = L_w - 20 \log r - 11 - \alpha r

mit:

 als frequenzabhängigem Absorptionskoeffizienten (typisch  dB/km).


b) Bodenabsorption und Reflexionen

Böden absorbieren Schall unterschiedlich stark. Die Gesamtreflexion wird durch die Bodenimpedanz beschrieben. Für Reflexionen kann das Maekawa-Kriterium für Schallschatten genutzt werden:

\Delta L = -5 + 20 \log \left( \frac{\lambda}{h} \right)

mit:

 der Wellenlänge,

 der Höhe des Hindernisses.



---

3. Hindernisse: Reflexion, Beugung und Abschattung

a) Reflexionen von Gebäuden

Ein Hindernis (z. B. eine Hauswand) reflektiert Schall mit einem Reflexionskoeffizienten :

R = \left| \frac{Z_2 - Z_1}{Z_2 + Z_1} \right|^2

mit Impedanzen  (Luft) und  (Gebäude).

b) Beugung um Gebäude herum

Schall kann um Gebäude herumbeugen. Dies wird durch das Kirchhoffsche Beugungsintegral beschrieben oder einfacher durch:

\Delta L = 10 \log \left( 1 + \frac{h}{\lambda} \right)

c) Mehrfache Reflexionen in Straßenschluchten

In engen Straßen kann Schall mehrfach reflektiert werden. Ein einfaches Modell ist:

L_p = L_w - 20 \log r + n R

mit  als Anzahl der Reflexionen.


---

4. Berechnung des Gesamtschallpegels in der Stadt

Jeder dieser Effekte verändert den Pegel  in jedem Punkt der Stadt. Eine Möglichkeit, die Schallverteilung flächendeckend zu berechnen, ist:

a) Rasterbasierte Simulation

Teile die Stadt in ein Gitter aus Punkten auf (z. B. alle 10 m ein Messpunkt).

Berechne für jeden Punkt den Schalldruckpegel unter Berücksichtigung von Dämpfung, Reflexionen und Beugung.


b) Ray-Tracing

Für präzisere Berechnungen kannst du Ray-Tracing für Schallwellen verwenden:

Schallstrahlen werden von der Quelle ausgesendet.

An jedem Hindernis wird geprüft, ob Reflexion, Beugung oder Transmission stattfindet.

Der Pegel am Zielort ergibt sich aus der Summe aller eintreffenden Strahlen.


c) Finite-Elemente-Methoden (FEM) oder Boundary Element Methods (BEM)

FEM für komplexe Strukturen mit variierender Impedanz.

BEM für Reflexionen und Beugungseffekte.



---

5. Zusammenfassung: Wie du die Schallverteilung berechnen kannst

1. Definiere die Schallquelle (z. B. Punktquelle mit ).


2. Ermittle die Dämpfung durch Luft, Boden, Hindernisse.


3. Berücksichtige Reflexion, Beugung und Abschattung.


4. Erstelle eine Gitterkarte oder nutze Ray-Tracing.


5. Summiere die Pegel aus allen Wegen auf.



Falls du ein konkretes Simulationsszenario hast (z. B. eine Karte einer Stadt mit einer Schallquelle), kann ich detaillierter auf die Implementierung in Python oder mit Software wie CadnaA oder Odeon eingehen
```


```
Beim Benchmarking und der Validierung generierter Bilder – vor allem in einem Szenario, in dem du physikalisch basierte Schallausbreitungsfelder auf Satellitenbildern erzeugst – stehst du vor der Herausforderung, dass du für neu generierte Bilder (die nicht exakt einem „Original“ entsprechen) keine direkten Paare zum Vergleich hast. Dennoch gibt es mehrere Strategien, um die Qualität und physikalische Konsistenz solcher Bilder zu bewerten und einen Benchmark aufzubauen.

---

## 1. Bewertung ohne direkte Ground-Truth-Paare

### a) Distributionsbasierte Metriken

Wenn du keine paarweise Ground-Truth-Referenz hast, kannst du statistische Abstände zwischen der Verteilung deiner generierten Bilder und der Verteilung der originalen (Trainings- oder Test-) Daten messen:

- *Fréchet Inception Distance (FID):*  
  Diese Metrik vergleicht die Verteilungen von Merkmalsvektoren (aus einem vortrainierten Netzwerk) der generierten Bilder mit denen der realen Bilder. Ein niedriger FID-Wert deutet darauf hin, dass die generierten Bilder der Zielverteilung nahekommen.

- *Kernel Inception Distance (KID):*  
  Ähnlich wie FID, aber mit einigen statistischen Vorteilen, vor allem bei kleineren Datensätzen.

- *Inception Score (IS):*  
  Dieser misst, wie gut ein Klassifikator (z. B. Inception-Netz) die generierten Bilder in unterschiedliche Klassen einordnen kann und ob diese Klassifizierungen mit hoher Sicherheit erfolgen. Allerdings ist IS oft weniger aussagekräftig, wenn die Domäne (hier z. B. Satellitenbilder mit Schallfeldinformationen) nicht gut mit dem vortrainierten Netz abgedeckt ist.

Diese Metriken erlauben es, die Verteilung der generierten Bilder mit der des echten Datensatzes zu vergleichen, ohne dass für jedes generierte Bild ein direktes Gegenstück existieren muss.

### b) Physikalisch informierte Metriken

Da es bei Schallausbreitungsmodellen auch um die Einhaltung physikalischer Gesetzmäßigkeiten geht, kannst du zusätzlich:

- *Energieverteilungen und Dämpfungsgesetze prüfen:*  
  Analysiere, ob die Abnahme des Schallpegels (z. B. mit der Entfernung von der Quelle) in den generierten Bildern mit den erwarteten physikalischen Modellen (wie z. B. \(L_p(r) = L_w - 20 \log_{10}(r) - K\,r\)) übereinstimmt.

- *Räumliche Muster und Schallschatten:*  
  Vergleiche charakteristische Strukturen wie Hotspots, Schallschatten hinter Gebäuden oder Reflexionsmuster. Hierzu kannst du beispielsweise die Korrelation zwischen simulierten „physikalischen Kennzahlen“ (z. B. gemessene Dämpfung an definierten Punkten) und den entsprechenden Regionen in den generierten Bildern berechnen.

- *Energieerhaltung:*  
  In einem abgeschlossenen System sollten gewisse Energieinvarianten (nach Abzug der Dämpfung) eingehalten sein. Diesen Aspekt kannst du mit geeigneten Integrationsmethoden über das Bildgebiet überprüfen.

### c) Experten- und Nutzerevaluation

- *Fachliche Beurteilung:*  
  Akustikexperten oder Ingenieure können die generierten Bilder qualitativ bewerten, um zu prüfen, ob die dargestellten Schallfelder physikalisch sinnvoll sind.

- *Crowdsourcing:*  
  In manchen Fällen können auch strukturierte Nutzertests (mit klar definierten Kriterien) dabei helfen, die wahrgenommene Qualität der generierten Bilder zu bewerten.

---

## 2. Aufbau eines generellen Benchmarks für generative Bildmodelle

Auch wenn für jeden generierten Fall kein direktes Ground-Truth-Bild existiert, kann man einen Benchmark folgendermaßen gestalten:

### a) Referenzdatensatz definieren

- *Trainings- und Testdatensatz:*  
  Teile deinen vorhandenen Datensatz (bestehend aus Satellitenbildern und den zugehörigen simulierten Schallausbreitungsfeldern) in Trainings- und Testdaten auf. Für die Testdaten hast du dann zumindest einen bekannten Referenzdatensatz, der als statistischer Vergleichsmaßstab dient.

- *Simulationsbasierte Ground Truth:*  
  Nutze aufwändige physikalische Simulationen (z. B. mit FEM, FDTD oder Ray-Tracing), um für ausgewählte Szenarien besonders hochwertige Referenzbilder zu erstellen. Diese können als „Ground Truth“ für die Testfälle dienen.

### b) Standardisierte Evaluationsprotokolle

- *Input-Output-Paare:*  
  Definiere klar, welche Eingabedaten (z. B. Satellitenbild, Quellparameter, Gebäudedaten) in den Benchmark einfließen. Für jeden dieser Eingaben gibt es entweder ein Referenzausgabebild (falls vorhanden) oder es wird erwartet, dass die generierten Bilder bestimmte physikalische Eigenschaften aufweisen.

- *Metriken und Skripte:*  
  Erstelle ein Evaluierungsskript (zum Beispiel in Python), das automatisch:
  - Distributionsbasierte Metriken (FID, KID) berechnet.
  - Physikalische Kennzahlen (z. B. Abnahme des Schalldruckpegels, Schallschatten, Interferenzmuster) extrahiert und vergleicht.
  - Eventuell auch menschliche Bewertungen integriert (zum Beispiel über eine standardisierte Umfrage).

### c) Benchmark-Framework und Vergleichsstudien

- *Open-Source Benchmark:*  
  Stelle den Benchmark als Open-Source-Paket zur Verfügung, sodass andere Modelle auf derselben Basis evaluiert werden können. Dies umfasst:
  - Den Datensatz (oder zumindest den Zugang zu simulierten Referenzfällen).
  - Evaluationsskripte und -protokolle.
  - Dokumentation der verwendeten Metriken und physikalischen Kriterien.

- *Vergleich verschiedener Modelle:*  
  Ermögliche den Vergleich von unterschiedlichen generativen Modellen (etwa GANs, VAEs, PINNs etc.), indem du standardisierte Kennzahlen und Testfälle definierst. So können etwa Metriken wie FID mit physikalisch informierten Fehlermaßen in einer Tabelle zusammengeführt werden.

---

## Zusammenfassung

1. *Validierung ohne direkte Ground Truth:*  
   Setze verteilungsbasierte Metriken (z. B. FID, KID) und physikalisch informierte Metriken ein, um zu beurteilen, ob die generierten Bilder sowohl optisch als auch physikalisch plausibel sind.

2. *Benchmark erstellen:*  
   – Definiere einen Referenzdatensatz (unter Verwendung von Testfällen und simulationsbasierten Ground-Truth-Bildern).  
   – Lege standardisierte Evaluationsprotokolle und Metriken fest, die auch physikalische Eigenschaften berücksichtigen.  
   – Erstelle ein Open-Source-Framework, das es erlaubt, verschiedene generative Modelle unter gleichen Bedingungen zu vergleichen.

Auch wenn bei der Generierung neuer Bilder kein direkter Vergleich zu einem einzelnen Originalbild möglich ist, erlaubt dieser mehrschichtige Ansatz die Beurteilung der Qualität und physikalischen Konsistenz der generierten Ergebnisse.
```

```
Ja, es gibt supervised Image-to-Image GANs ohne U-Net. U-Net ist besonders für Bild-zu-Bild-Übersetzung nützlich, aber es gibt alternativen Generator-Architekturen, die ohne U-Net funktionieren.


---

1. ResNet-basierte Generatoren (ohne U-Net)

Ersetzt U-Net durch ResNet-Blöcke für tiefere und stabilere Netzwerke.

Beispiel:

Pix2Pix mit ResNet statt U-Net

CycleGAN (nutzt ResNet-Generator, kein U-Net!)


Vorteile:
✔ Funktioniert besser für Stiltransfer & Texturänderungen
✔ Reduziert Artefakte im Vergleich zu U-Net
✔ Besser für natürliche Bilder mit weichen Übergängen

Nachteile:
❌ Mehr Parameter, kann schwieriger zu trainieren sein


✅ Einsatzgebiete:
✔ Foto → Malerei
✔ Low-Res → High-Res


---

2. Transformer-basierte GANs für Image-to-Image

Ersetzt CNNs durch Vision Transformers (ViTs)

Beispiel:

TransGAN (GAN mit reinem Transformer)

Image-to-Image mit GANs + Swin Transformer


Vorteile:
✔ Besser für langfristige Abhängigkeiten im Bild
✔ Weniger lokale Verzerrungen als CNNs

Nachteile:
❌ Sehr rechenintensiv


✅ Einsatzgebiete:
✔ Texturtransfer
✔ Super-Resolution


---

3. StyleGAN für Bild-zu-Bild (StyleGAN-NADA, StyleGAN-XL)

Verwendet Mapping-Netzwerke & Adaptive Instance Normalization (AdaIN)

Beispiel:

StyleGAN-NADA (Text- oder Bild-basierte Steuerung)

StyleGAN-XL (hochauflösende Generierung)


Vorteile:
✔ Extrem hochwertige generierte Bilder
✔ Sehr kontrollierbar durch latenten Raum

Nachteile:
❌ Braucht viele Daten & lange Trainingszeit


✅ Einsatzgebiete:
✔ Gesichts- oder Stiltransformation


---

4. GANs mit Perceptual Loss statt U-Net

Ersetzt klassische GAN-Verluste durch Deep Feature Loss aus einem vortrainierten Netz (z. B. VGG16)

Beispiel:

SRGAN (Super-Resolution GAN, nutzt VGG-Features)

Perceptual Pix2Pix (nutzt VGG statt U-Net)


Vorteile:
✔ Nutzt semantische Merkmale statt reine Pixelvergleiche
✔ Führt zu besseren Details und natürlicheren Texturen

Nachteile:
❌ Schwieriger zu trainieren als Standard-GANs


✅ Einsatzgebiete:
✔ Super-Resolution
✔ Realistische Texturen


---

Fazit: Welche GAN-Alternative ohne U-Net ist für dich geeignet?

Möchtest du eine PyTorch/TensorFlow-Implementierung für einen dieser GAN-Typen? 😊
```

```
Für supervised Bild-zu-Bild-Generierung gibt es mehrere generative Modelle, die speziell darauf ausgelegt sind, ein Eingangsbild in ein Ausgangsbild zu transformieren. Hier sind die wichtigsten Methoden:


---

1. Conditional GANs (cGANs) → Pix2Pix

Prinzip: Erweiterung eines GANs, bei dem der Generator ein U-Net ist und der Diskriminator zwischen echten und generierten Bildern unterscheidet.

Beispiel:

Pix2Pix → z. B. für Schwarz-Weiß-zu-Farbbilder

CycleGAN → wenn kein 1:1 Mapping zwischen Bildern existiert


Supervision: Voll-supervised (benötigt Eingangs- und Zielbild)

Vorteile: Erzeugt detaillierte und realistische Bilder, funktioniert gut mit kleinen Datensätzen.

Nachteil: Kann Artefakte erzeugen.


✅ Anwendungsfälle:
✔ Skizze → Foto
✔ Tag → Nacht
✔ Tiefenbild → RGB


---

2. Diffusionsmodelle für Image-to-Image-Translation

Prinzip: Statt direkt ein Bild zu erzeugen, lernt das Modell, schrittweise Rauschen aus Bildern zu entfernen.

Beispiel:

Stable Diffusion Inpainting → kann Bilder ergänzen oder verändern

ControlNet (Stable Diffusion) → erlaubt gezielte Kontrolle über die Bildtransformation (z. B. Kantenbilder in Fotos umwandeln)


Supervision: Supervised oder Weakly-Supervised

Vorteile: Hochwertige und natürliche Bilder, weniger Artefakte als GANs

Nachteil: Langsam, benötigt viele Rechenressourcen


✅ Anwendungsfälle:
✔ Kantenbilder → Foto
✔ Pose (Skelett) → Mensch
✔ Kartendaten → Satellitenbild


---

3. Autoregressive Modelle (ImageGPT, Parti, DALL·E)

Prinzip: Bilder werden Pixel für Pixel oder Patch für Patch vorhergesagt, ähnlich wie GPT für Texte.

Beispiel:

ImageGPT von OpenAI

DALL·E 2 (mit Inpainting-Funktion)

Parti (Google)


Supervision: Voll-supervised

Vorteile: Sehr flexibel, kann Text- oder Bild-Prompts nutzen

Nachteil: Langsame Generierung, benötigt viele Trainingsdaten


✅ Anwendungsfälle:
✔ Hochskalierung & Super-Resolution
✔ Inpainting & Bildvervollständigung


---

4. Normalizing Flows (Glow, RealNVP) für Bild-zu-Bild-Generierung

Prinzip: Invertierbare neuronale Netze, die ein Bild in eine latente Repräsentation umwandeln und wieder zurückführen können.

Beispiel:

Glow (OpenAI) – generiert Gesichter und kann diese manipulieren

RealNVP – für kontrollierte Bildtransformation


Supervision: Semi-supervised

Vorteile: Sehr interpretiert, kann hochauflösende Bilder erzeugen

Nachteil: Nicht so gut für kreative Generierung


✅ Anwendungsfälle:
✔ Gesichtsmanipulation
✔ Style Transfer mit hoher Kontrolle


---

Welche Methode ist für dich am besten?

Möchtest du eine Pix2Pix- oder Diffusionsmodell-Implementierung mit TensorFlow/PyTorch sehen? 😊
```


```
Ja, das könnte durchaus helfen. Dein generatives Modell lernt eine Abbildung von Bilddaten auf die Schallausbreitung. Wenn du zusätzliche physikalische Daten hinzufügst, die mit der Schallausbreitung korrelieren, kann das Modell eine reichhaltigere Repräsentation lernen und somit die Generalisierungsfähigkeit verbessern. Hier sind einige Ansätze:

1. Multi-Task Learning mit anderen physikalischen Phänomenen

Wärmeausbreitung: Schall und Wärme folgen in bestimmten Materialien ähnlichen Wellengleichungen. Ein Modell, das beides lernt, könnte verbesserte physikalische Intuition entwickeln.

Fluiddynamik (CFD-Simulationen): Besonders bei der Schallausbreitung in Gasen oder Flüssigkeiten könnten Strömungsfelder nützliche Zusatzinformationen liefern.

Elektromagnetische Wellen: Da Wellenmechaniken ähnliche mathematische Strukturen haben, könnte ein Modell, das auch Lichtbrechung oder Radarausbreitung lernt, für Schall robuster werden.


2. Auxiliary Inputs für zusätzliche Constraints

Materialeigenschaften (Dichte, Elastizitätsmodul, Impedanz): Wenn das Modell die physikalischen Eigenschaften der Umgebung kennt, kann es präzisere Vorhersagen über die Schallausbreitung treffen.

Geometrische Strukturen (Tiefenkarten, Oberflächenrauhigkeit): Ein Depth- oder Normal-Map könnte helfen, Reflexionen und Beugungseffekte besser zu modellieren.


3. Physik-informierte Verluste (PINNs)

Du könntest eine physikbasierte Verlustfunktion hinzufügen, die z. B. die Wellengleichung als Constraint einbindet. Dadurch kann das Modell physikalisch konsistentere Ergebnisse liefern.


4. Transfer Learning von anderen Wellenausbreitungsmodellen

Falls es bereits ein Modell gibt, das für ähnliche Aufgaben trainiert wurde (z. B. in der Akustik oder Radarverarbeitung), könntest du dessen Gewichte als Pretraining für dein Modell nutzen.


Zusammengefasst: Ja, durch Training mit anderen physikalischen Daten – insbesondere solchen mit ähnlichen Wellengleichungen – kannst du die Performance deines Modells steigern. Eine Kombination aus Multi-Task Learning, zusätzlichen physikalischen Inputs und physikbasierten Verlustfunktionen könnte besonders effektiv sein.
```


```
Ja, du kannst generative Modelle, insbesondere GANs, in einem stärker überwachten (supervised) Setting verwenden. Klassische GANs sind unüberwacht, da sie versuchen, Daten zu modellieren, ohne explizite Labels zu nutzen. Es gibt jedoch mehrere Möglichkeiten, sie für eine supervised Aufgabe wie Image-to-Image Translation zu verwenden.

## *1. Supervised Image-to-Image Translation mit GANs*
Ein bekanntes Beispiel ist *Pix2Pix*, das eine U-Net-ähnliche Architektur für den Generator verwendet und mit gepaarten Bildern trainiert wird (z. B. eine Skizze als Input und ein fotorealistisches Bild als Output).

In PyTorch kannst du das so umsetzen:

### *Schritt 1: Lade benötigte Bibliotheken*
python
import torch
import torch.nn as nn
import torch.optim as optim
from …
Hier ist die Umsetzung eines *supervised GANs (Pix2Pix)* mit *TensorFlow und Keras*. Es nimmt ein Input-Bild und generiert ein entsprechendes Output-Bild.  

---

## *1. Bibliotheken importieren*  
python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


---

## *2. Generator (U-Net Architektur)*  
Der Generator erhält ein Bild als Eingabe und gibt ein transformiertes Bild aus.  
python
def build_generator():
    inputs = keras.Input(shape=(256, 256, 3))

    # Encoder
    down1 = layers.Conv2D(64, (4, 4), strides=2, padding="same", activation="relu")(inputs)
    down2 = layers.Conv2D(128, (4, 4), strides=2, padding="same", activation="relu")(down1)
    
    # Bottleneck
    bottleneck = layers.Conv2D(256, (4, 4), strides=2, padding="same", activation="relu")(down2)
    
    # Decoder
    up1 = layers.Conv2DTranspose(128, (4, 4), strides=2, padding="same", activation="relu")(bottleneck)
    up2 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding="same", activation="relu")(up1)
    outputs = layers.Conv2DTranspose(3, (4, 4), strides=2, padding="same", activation="tanh")(up2)

    return keras.Model(inputs, outputs, name="Generator")

generator = build_generator()
generator.summary()


---

## *3. Diskriminator (CNN-Klassifikator)*  
Der Diskriminator unterscheidet zwischen echten (Input+Ziel) und generierten (Input+Fake) Bildpaaren.  
python
def build_discriminator():
    input_image = keras.Input(shape=(256, 256, 3))
    target_image = keras.Input(shape=(256, 256, 3))

    combined = layers.Concatenate()([input_image, target_image])

    conv1 = layers.Conv2D(64, (4, 4), strides=2, padding="same", activation="relu")(combined)
    conv2 = layers.Conv2D(128, (4, 4), strides=2, padding="same", activation="relu")(conv1)
    conv3 = layers.Conv2D(256, (4, 4), strides=2, padding="same", activation="relu")(conv2)

    flatten = layers.Flatten()(conv3)
    output = layers.Dense(1, activation="sigmoid")(flatten)

    return keras.Model([input_image, target_image], output, name="Discriminator")

discriminator = build_discriminator()
discriminator.summary()


---

## *4. Verlustfunktionen & Optimizer*  
Das Training nutzt *Binary Cross-Entropy für GAN-Loss* und *L1-Loss für den Generator*.  
python
loss_object = keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output, generated_image, target_image):
    gan_loss = loss_object(tf.ones_like(fake_output), fake_output)
    l1_loss = tf.reduce_mean(tf.abs(target_image - generated_image))
    return gan_loss + 100 * l1_loss  # L1-Loss gewichtet

def discriminator_loss(real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    return (real_loss + fake_loss) / 2

generator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)


---

## *5. Training Loop*
python
@tf.function
def train_step(input_image, target_image):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(input_image, training=True)

        real_output = discriminator([input_image, target_image], training=True)
        fake_output = discriminator([input_image, generated_image], training=True)

        gen_loss = generator_loss(fake_output, generated_image, target_image)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


---

## *6. Training starten*
Hier wird ein Dataset simuliert (du kannst dein eigenes verwenden).  
python
num_epochs = 100

for epoch in range(num_epochs):
    input_image = np.random.rand(1, 256, 256, 3).astype("float32")  # Simuliertes Input-Bild
    target_image = np.random.rand(1, 256, 256, 3).astype("float32")  # Simuliertes Target-Bild

    g_loss, d_loss = train_step(input_image, target_image)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Generator Loss: {g_loss.numpy()}, Discriminator Loss: {d_loss.numpy()}")


---

## *Fazit*
- *GANs können supervised genutzt werden, indem sie mit gepaarten Bildern trainiert werden.*
- *Hier haben wir ein einfaches Pix2Pix mit TensorFlow gebaut.*
- *Der Generator nutzt U-Net, und der Diskriminator prüft, ob ein generiertes Bild realistisch ist.*  

Falls du dein eigenes Dataset verwenden willst (z. B. aus Bildern), kannst du tf.data.Dataset nutzen. Sag mir Bescheid, falls du es anpassen möchtest! 🚀
```


DeepSeek:

```
Um die Schallausbreitung in städtischen Umgebungen auf Satellitenbildern mit generativen Modellen darzustellen und einen Benchmark zu erstellen, können Sie wie folgt vorgehen:

---

### **1. Geeignete generative Modelle**
Folgende Modelle kommen für die Aufgabe infrage, jeweils mit Vor- und Nachteilen:

#### **a) Conditional GANs (cGANs)**
- **Beispiele**: Pix2Pix, SPADE (GaNyu), CycleGAN (für ungepaarte Daten).
- **Stärken**: 
  - Gut für Bild-zu-Bild-Übersetzung (z. B. Satellitenbild → Schallausbreitung).
  - Erzeugt hochauflösende, detaillierte Ergebnisse.
- **Schwächen**: 
  - Instabiles Training, erfordert große Datensätze.
  - Benötigt ggf. gepaarte Daten (außer CycleGAN).

#### **b) Diffusion Models**
- **Beispiele**: DDPM (Denoising Diffusion Probabilistic Models), Stable Diffusion.
- **Stärken**: 
  - Hohe Qualität und Diversität der generierten Bilder.
  - Stabileres Training als GANs.
- **Schwächen**: 
  - Rechenintensiv (lange Trainings- und Inferenzzeiten).

#### **c) Hybridmodelle mit physikalischen Constraints**
- **Ansatz**: Kombination von generativen Modellen mit physikalischen Gleichungen (z. B. Integration der Wellengleichung in den Loss).
- **Stärken**: 
  - Erzwingt physikalisch plausible Ergebnisse.
  - Reduziert Halluzinationen.
- **Schwächen**: 
  - Komplexe Implementierung, erfordert Domänenwissen.

#### **d) Vision Transformer (ViT)**
- **Beispiele**: ViT, Swin Transformer.
- **Stärken**: 
  - Skalierbar für große Datensätze.
  - Erfasst globale Kontexte (z. B. städtische Strukturen).
- **Schwächen**: 
  - Erfordert viel Datenvorbereitung und Rechenleistung.

---

### **2. Benchmark-Erstellung**
Ein aussagekräftiger Benchmark umfasst folgende Schritte:

#### **a) Datensatzvorbereitung**
- **Alignment**: Sicherstellen, dass Satellitenbilder und Schalldaten pixelgenau übereinstimmen.
- **Aufteilung**: Trainings- (70%), Validierungs- (15%) und Testdaten (15%).
- **Augmentierung**: Spiegelungen, Rotationen, Rauschen, um Überanpassung zu vermeiden.

#### **b) Metriken für die Evaluation**
- **Pixel-basierte Metriken**:
  - **PSNR/SSIM**: Misst strukturelle Ähnlichkeit zwischen generierten und Ground-Truth-Bildern.
  - **MSE (Mean Squared Error)**: Quantifiziert pixelweise Abweichungen.
- **Feature-basierte Metriken**:
  - **FID (Fréchet Inception Distance)**: Bewertet die visuelle Qualität.
  - **Kernel MMD**: Misst Verteilungsähnlichkeit.
- **Physikalische Metriken**:
  - **Wellengleichungs-Konformität**: Validierung durch Simulation (z. B. Vergleich mit COMSOL/ANSYS).
  - **Energieerhaltung**: Prüfung, ob die Schallintensität physikalisch plausibel abnimmt.

#### **c) Baseline-Modelle**
- **Einfache Architekturen**: UNet, ResNet als Baseline für Bild-zu-Bild-Übersetzung.
- **Physik-basierte Modelle**: Analytische Lösungen oder vereinfachte Simulationen (z. B. ray tracing für Schall).

#### **d) Experiment-Design**
1. **Hyperparameter-Tuning**: Für jedes Modell optimale Lernraten, Batch-Größen etc. ermitteln.
2. **Reproduzierbarkeit**: Fixierte Random Seeds, öffentlicher Code (z. B. GitHub).
3. **Vergleich**: Jedes Modell wird auf dem gleichen Testset mit allen Metriken evaluiert.
4. **Qualitative Analyse**: Visuelle Inspektion durch Experten (z. B. Akustiker:innen).

#### **e) Beispiel-Pipeline**
python
# Pseudocode für Benchmarking
for model in [pix2pix, diffusion_model, hybrid_model]:
    predictions = model.predict(test_images)
    calculate_metrics(predictions, ground_truth)
    run_physics_validation(predictions)
    save_results(model, metrics)

---

### **3. Praktische Empfehlungen**
- **Start mit Pix2Pix**: Einfach zu implementieren, gut dokumentiert.
- **Datenqualität priorisieren**: Fehlausrichtungen zwischen Bildern und Schalldaten ruinieren das Training.
- **Ressourcenabschätzung**: Diffusion Models erfordern GPUs mit >16 GB VRAM; GANs sind leichter.
- **Open-Source-Tools**: 
  - **GANs**: PyTorch Lightning, MMGeneration.
  - **Diffusion**: Hugging Face Diffusers.
  - **Physikalische Validierung**: FEniCS (PDE-Lösungen).

---

### **4. Zusammenfassung**
- **Modelle**: Conditional GANs (Pix2Pix), Diffusion Models, Hybridansätze.
- **Benchmark**: Kombination aus pixelbasierten Metriken, FID und physikalischer Validierung.
- **Entscheidungshilfe**: Nutzen Sie den Benchmark, um abzuwägen, ob komplexe Modelle (z. B. Diffusion) den Mehrwert gegenüber schnelleren Modellen (z. B. GANs) rechtfertigen.
```


```
Wenn Sie neue Bilder generieren möchten, die die Schallausbreitung in städtischen Umgebungen darstellen, und nicht nur bestehende Satellitenbilder in Schallausbreitungsbilder übersetzen, dann benötigen Sie **generative Modelle, die neue Daten synthetisieren können**. Hier sind die geeigneten Modelle und Ansätze:

---

### **1. Geeignete generative Modelle**
#### **a) Generative Adversarial Networks (GANs)**
- **Beispiele**: StyleGAN, BigGAN, ProGAN.
- **Stärken**:
  - Erzeugt hochauflösende, realistische Bilder.
  - Kann komplexe Muster (z. B. städtische Strukturen) lernen.
- **Schwächen**:
  - Instabiles Training, erfordert große Datensätze.
  - Schwierig, physikalische Constraints zu integrieren.

#### **b) Variational Autoencoder (VAE)**
- **Beispiele**: VQ-VAE, NVAE.
- **Stärken**:
  - Stabileres Training als GANs.
  - Kann latenten Raum für Interpolationen nutzen.
- **Schwächen**:
  - Oft weniger scharfe Bilder als GANs.
  - Benötigt sorgfältige Modellierung des latenten Raums.

#### **c) Diffusion Models**
- **Beispiele**: DDPM, Stable Diffusion.
- **Stärken**:
  - Hohe Qualität und Diversität der generierten Bilder.
  - Stabileres Training als GANs.
- **Schwächen**:
  - Rechenintensiv (lange Trainings- und Inferenzzeiten).

#### **d) Transformer-basierte Modelle**
- **Beispiele**: DALL·E, ImageGPT.
- **Stärken**:
  - Skalierbar für große Datensätze.
  - Erfasst globale Kontexte (z. B. städtische Strukturen).
- **Schwächen**:
  - Erfordert viel Datenvorbereitung und Rechenleistung.

#### **e) Physikalisch-inspirierte generative Modelle**
- **Ansatz**: Kombination von generativen Modellen mit physikalischen Gleichungen (z. B. Integration der Wellengleichung in den Loss).
- **Stärken**:
  - Erzwingt physikalisch plausible Ergebnisse.
  - Reduziert Halluzinationen.
- **Schwächen**:
  - Komplexe Implementierung, erfordert Domänenwissen.

---

### **2. Benchmark-Erstellung**
Ein Benchmark für generative Modelle umfasst folgende Schritte:

#### **a) Datensatzvorbereitung**
- **Datenqualität**: Sicherstellen, dass der Datensatz repräsentativ und vielfältig ist.
- **Aufteilung**: Trainings- (80%), Validierungs- (10%) und Testdaten (10%).
- **Augmentierung**: Spiegelungen, Rotationen, Rauschen, um Überanpassung zu vermeiden.

#### **b) Metriken für die Evaluation**
- **Qualitätsmetriken**:
  - **FID (Fréchet Inception Distance)**: Bewertet die visuelle Qualität.
  - **IS (Inception Score)**: Misst die Diversität und Qualität der generierten Bilder.
- **Diversitätsmetriken**:
  - **Precision/Recall**: Misst, wie gut die generierten Bilder die Trainingsdaten abdecken.
  - **Kernel MMD**: Misst Verteilungsähnlichkeit.
- **Physikalische Metriken**:
  - **Wellengleichungs-Konformität**: Validierung durch Simulation (z. B. Vergleich mit COMSOL/ANSYS).
  - **Energieerhaltung**: Prüfung, ob die Schallintensität physikalisch plausibel abnimmt.

#### **c) Baseline-Modelle**
- **Einfache Architekturen**: VAE, GAN (z. B. DCGAN) als Baseline.
- **Physik-basierte Modelle**: Analytische Lösungen oder vereinfachte Simulationen (z. B. ray tracing für Schall).

#### **d) Experiment-Design**
1. **Hyperparameter-Tuning**: Für jedes Modell optimale Lernraten, Batch-Größen etc. ermitteln.
2. **Reproduzierbarkeit**: Fixierte Random Seeds, öffentlicher Code (z. B. GitHub).
3. **Vergleich**: Jedes Modell wird auf dem gleichen Testset mit allen Metriken evaluiert.
4. **Qualitative Analyse**: Visuelle Inspektion durch Experten (z. B. Akustiker:innen).

#### **e) Beispiel-Pipeline**
python
# Pseudocode für Benchmarking
for model in [stylegan, diffusion_model, vae]:
    generated_images = model.generate(num_samples=1000)
    calculate_metrics(generated_images, training_data)
    run_physics_validation(generated_images)
    save_results(model, metrics)

---

### **3. Praktische Empfehlungen**
- **Start mit GANs oder VAEs**: Einfach zu implementieren, gut dokumentiert.
- **Datenqualität priorisieren**: Fehlende Vielfalt im Datensatz führt zu schlechten Ergebnissen.
- **Ressourcenabschätzung**: Diffusion Models erfordern GPUs mit >16 GB VRAM; GANs sind leichter.
- **Open-Source-Tools**: 
  - **GANs**: PyTorch Lightning, MMGeneration.
  - **Diffusion**: Hugging Face Diffusers.
  - **Physikalische Validierung**: FEniCS (PDE-Lösungen).

---

### **4. Zusammenfassung**
- **Modelle**: GANs (StyleGAN), Diffusion Models, VAEs, Transformer-basierte Modelle.
- **Benchmark**: Kombination aus Qualitätsmetriken (FID, IS), Diversitätsmetriken (Precision/Recall) und physikalischer Validierung.
- **Entscheidungshilfe**: Nutzen Sie den Benchmark, um abzuwägen, ob komplexe Modelle (z. B. Diffusion) den Mehrwert gegenüber schnelleren Modellen (z. B. GANs) rechtfertigen.

Falls Sie spezifische Fragen zur Implementierung oder zum Training haben, lassen Sie es mich wissen!
```


```
Um die Schallausbreitung in städtischen Umgebungen mathematisch zu beschreiben, müssen Sie die **physikalischen Prinzipien der Akustik** und deren Interaktion mit Hindernissen (z. B. Häusern) modellieren. Die Schallausbreitung wird durch die **Wellengleichung** beschrieben, und die Interaktion mit Hindernissen kann durch Randbedingungen und Reflexionsgesetze erfasst werden. Hier ist eine detaillierte Beschreibung:

---

### **1. Mathematische Beschreibung der Schallausbreitung**
#### **a) Wellengleichung**
Die Schallausbreitung in einem Medium (z. B. Luft) wird durch die **akustische Wellengleichung** beschrieben:
\[
\frac{\partial^2 p}{\partial t^2} = c^2 \nabla^2 p,
\]
wobei:
- \( p(\mathbf{x}, t) \) der Schalldruck am Ort \( \mathbf{x} \) zur Zeit \( t \) ist,
- \( c \) die Schallgeschwindigkeit im Medium ist,
- \( \nabla^2 \) der Laplace-Operator ist.

#### **b) Randbedingungen an Hindernissen**
Häuser und andere Hindernisse beeinflussen die Schallausbreitung durch:
1. **Reflexion**: Schallwellen werden an Oberflächen reflektiert. Der Reflexionsgrad hängt von den Materialeigenschaften ab.
2. **Absorption**: Ein Teil der Schallenergie wird von Hindernissen absorbiert.
3. **Beugung**: Schallwellen können um Hindernisse herum gebeugt werden.
4. **Streuung**: Unebenheiten an Oberflächen streuen den Schall.

Diese Effekte können durch Randbedingungen modelliert werden:
- **Reflexion**: \( \frac{\partial p}{\partial n} = 0 \) (harte Wand) oder \( p = 0 \) (weiche Wand).
- **Absorption**: \( \frac{\partial p}{\partial n} + \alpha p = 0 \), wobei \( \alpha \) der Absorptionskoeffizient ist.

#### **c) Dämpfung**
Schallwellen werden durch die Luft gedämpft. Dies kann durch einen Dämpfungsterm in der Wellengleichung beschrieben werden:
\[
\frac{\partial^2 p}{\partial t^2} + \beta \frac{\partial p}{\partial t} = c^2 \nabla^2 p,
\]
wobei \( \beta \) der Dämpfungskoeffizient ist.

---

### **2. Wahre Dichtefunktion (True Density Function)**
Die **wahre Dichtefunktion**, die die Modelle lernen müssen, beschreibt die Wahrscheinlichkeitsverteilung der Schallausbreitung in Abhängigkeit von:
- **Geometrie der Umgebung** (Position und Form der Häuser),
- **Materialeigenschaften** (Reflexion, Absorption),
- **Schallquellen** (Position, Frequenz, Intensität).

Mathematisch kann dies als **bedingte Wahrscheinlichkeitsverteilung** formuliert werden:
\[
P(p(\mathbf{x}, t) \,|\, \text{Geometrie, Materialeigenschaften, Schallquellen}),
\]
wobei \( p(\mathbf{x}, t) \) der Schalldruck ist.

---

### **3. Vereinfachte Modelle für die Interaktion mit Häusern**
#### **a) Ray Tracing**
- **Ansatz**: Schall wird als Strahlen modelliert, die sich geradlinig ausbreiten und an Hindernissen reflektiert werden.
- **Vorteil**: Einfach zu implementieren, gut für hohe Frequenzen.
- **Nachteil**: Ignoriert Beugung und Interferenz.

#### **b) Geometrische Akustik**
- **Ansatz**: Kombiniert Ray Tracing mit Absorptions- und Streueffekten.
- **Vorteil**: Realistischere Ergebnisse als reines Ray Tracing.
- **Nachteil**: Rechenintensiv.

#### **c) Finite-Elemente-Methode (FEM)**
- **Ansatz**: Numerische Lösung der Wellengleichung mit Randbedingungen.
- **Vorteil**: Sehr genau, berücksichtigt alle physikalischen Effekte.
- **Nachteil**: Extrem rechenintensiv, nicht für Echtzeitanwendungen geeignet.

---

### **4. Integration in generative Modelle**
Generative Modelle können die Schallausbreitung implizit oder explizit lernen:

#### **a) Implizites Lernen**
- **Ansatz**: Das Modell lernt die Dichtefunktion \( P(p(\mathbf{x}, t) \,|\, \text{Geometrie, Materialeigenschaften, Schallquellen}) \) direkt aus den Daten.
- **Beispiele**: GANs, VAEs, Diffusion Models.
- **Vorteil**: Keine explizite Modellierung der Physik erforderlich.
- **Nachteil**: Erfordert große, repräsentative Datensätze.

#### **b) Explizites Lernen**
- **Ansatz**: Physikalische Gleichungen (z. B. Wellengleichung) werden in den Loss integriert.
- **Beispiele**: Physically-Informed Neural Networks (PINNs), Hybridmodelle.
- **Vorteil**: Erzwingt physikalisch plausible Ergebnisse.
- **Nachteil**: Komplexe Implementierung, erfordert Domänenwissen.

---

### **5. Beispiel: Physically-Informed Neural Networks (PINNs)**
PINNs kombinieren neuronale Netze mit physikalischen Gleichungen. Der Loss besteht aus:
1. **Daten-Loss**: Fehler zwischen Vorhersage und gemessenen Daten.
2. **Physik-Loss**: Fehler in der Wellengleichung und Randbedingungen.

Mathematisch:
\[
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physik}},
\]
wobei \( \lambda \) ein Regularisierungsparameter ist.

---

### **6. Zusammenfassung**
- **Wahre Dichtefunktion**: \( P(p(\mathbf{x}, t) \,|\, \text{Geometrie, Materialeigenschaften, Schallquellen}) \).
- **Physikalische Grundlage**: Wellengleichung, Randbedingungen, Dämpfung.
- **Modelle**: GANs, VAEs, Diffusion Models (implizit) oder PINNs (explizit).
- **Benchmark**: Kombination aus physikalischen Metriken (Wellengleichungs-Konformität) und Qualitätsmetriken (FID, IS).

Falls Sie weitere Details zur Implementierung oder zu spezifischen Modellen benötigen, lassen Sie es mich wissen!
```


```
Um einen **physikalischen Loss** in ein generatives Modell zu integrieren, das Bilder der Schallausbreitung erzeugt, müssen Sie die physikalischen Prinzipien (z. B. die Wellengleichung und Randbedingungen) in den Trainingsprozess einbeziehen. Dies kann durch die Verwendung von **Physically-Informed Neural Networks (PINNs)** oder durch die Integration physikalischer Constraints in den Loss-Funktion erreicht werden. Hier ist eine Schritt-für-Schritt-Anleitung:

---

### **1. Grundidee**
Der **physikalische Loss** bestraft Abweichungen von den physikalischen Gesetzen (z. B. der Wellengleichung). Der Gesamt-Loss besteht aus:
1. **Daten-Loss**: Misst die Übereinstimmung mit den Trainingsdaten (z. B. pixelweiser Fehler).
2. **Physik-Loss**: Misst die Übereinstimmung mit den physikalischen Gesetzen.
3. **Regularisierung**: Optional, um Overfitting zu vermeiden.

Mathematisch:
\[
\mathcal{L}_{\text{gesamt}} = \mathcal{L}_{\text{Daten}} + \lambda \mathcal{L}_{\text{Physik}},
\]
wobei \( \lambda \) ein Hyperparameter ist, der die Gewichtung des physikalischen Loss steuert.

---

### **2. Physikalischer Loss für Schallausbreitung**
#### **a) Wellengleichung**
Die Wellengleichung lautet:
\[
\frac{\partial^2 p}{\partial t^2} = c^2 \nabla^2 p,
\]
wobei \( p(\mathbf{x}, t) \) der Schalldruck ist und \( c \) die Schallgeschwindigkeit.

Der **physikalische Loss** kann als mittlerer quadratischer Fehler (MSE) der Wellengleichung definiert werden:
\[
\mathcal{L}_{\text{Physik}} = \frac{1}{N} \sum_{i=1}^N \left( \frac{\partial^2 p_i}{\partial t^2} - c^2 \nabla^2 p_i \right)^2,
\]
wobei \( N \) die Anzahl der Punkte im Bild ist.

#### **b) Randbedingungen**
Randbedingungen (z. B. Reflexion an Häusern) können ebenfalls in den Loss integriert werden. Für eine harte Wand gilt:
\[
\frac{\partial p}{\partial n} = 0,
\]
wobei \( n \) die Normale zur Wand ist. Der Loss für Randbedingungen kann als:
\[
\mathcal{L}_{\text{Rand}} = \frac{1}{M} \sum_{j=1}^M \left( \frac{\partial p_j}{\partial n} \right)^2
\]
definiert werden, wobei \( M \) die Anzahl der Randpunkte ist.

#### **c) Gesamt-Loss**
Der Gesamt-Loss kombiniert Daten-Loss, physikalischen Loss und Randbedingungen:
\[
\mathcal{L}_{\text{gesamt}} = \mathcal{L}_{\text{Daten}} + \lambda_1 \mathcal{L}_{\text{Physik}} + \lambda_2 \mathcal{L}_{\text{Rand}}.
\]

---

### **3. Implementierung**
#### **a) Daten-Loss**
Der Daten-Loss kann ein pixelweiser Fehler (z. B. MSE) zwischen dem generierten Bild \( \hat{y} \) und dem Ground-Truth-Bild \( y \) sein:
\[
\mathcal{L}_{\text{Daten}} = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W (\hat{y}_{ij} - y_{ij})^2,
\]
wobei \( H \) und \( W \) die Höhe und Breite des Bildes sind.

#### **b) Physikalischer Loss**
1. **Berechnung der Ableitungen**:
   - Verwenden Sie finite Differenzen oder automatische Differentiation (z. B. in PyTorch oder TensorFlow), um die partiellen Ableitungen \( \frac{\partial^2 p}{\partial t^2} \) und \( \nabla^2 p \) zu berechnen.
   - Beispiel für \( \nabla^2 p \):
     \[
     \nabla^2 p \approx \frac{p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1} - 4p_{i,j}}{h^2},
     \]
     wobei \( h \) die Schrittweite ist.

2. **Berechnung des physikalischen Loss**:
   - Berechnen Sie den Fehler in der Wellengleichung für jeden Pixel.

#### **c) Randbedingungen**
- Identifizieren Sie die Pixel, die Hindernisse (z. B. Häuser) darstellen.
- Berechnen Sie den Fehler in den Randbedingungen für diese Pixel.

---

### **4. Beispielcode (PyTorch)**
Hier ist ein vereinfachtes Beispiel, wie Sie den physikalischen Loss in PyTorch implementieren können:

python
import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    def __init__(self, c, lambda_phys, lambda_boundary):
        super(PhysicsInformedLoss, self).__init__()
        self.c = c  # Schallgeschwindigkeit
        self.lambda_phys = lambda_phys  # Gewichtung des physikalischen Loss
        self.lambda_boundary = lambda_boundary  # Gewichtung der Randbedingungen

    def forward(self, predicted_image, ground_truth, boundary_mask):
        # Daten-Loss (MSE)
        data_loss = torch.mean((predicted_image - ground_truth) ** 2)

        # Physikalischer Loss (Wellengleichung)
        dx = 1.0  # Schrittweite in x-Richtung
        dy = 1.0  # Schrittweite in y-Richtung
        laplacian = (predicted_image[1:, :] + predicted_image[:-1, :] +
                     predicted_image[:, 1:] + predicted_image[:, :-1] -
                     4 * predicted_image[1:-1, 1:-1]) / (dx * dy)
        physics_loss = torch.mean((laplacian) ** 2)

        # Randbedingungen (harte Wand)
        boundary_loss = torch.mean((predicted_image[boundary_mask] - 0) ** 2)

        # Gesamt-Loss
        total_loss = (data_loss +
                      self.lambda_phys * physics_loss +
                      self.lambda_boundary * boundary_loss)
        return total_loss

# Beispiel: Modelltraining
model = ...  # Ihr generatives Modell
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = PhysicsInformedLoss(c=340, lambda_phys=1.0, lambda_boundary=1.0)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    generated_image = model(input_data)
    loss = loss_fn(generated_image, ground_truth, boundary_mask)
    loss.backward()
    optimizer.step()

---

### **5. Praktische Tipps**
- **Schrittweite**: Wählen Sie die Schrittweite \( h \) sorgfältig, um numerische Instabilitäten zu vermeiden.
- **Gewichtung**: Passen Sie \( \lambda_1 \) und \( \lambda_2 \) an, um das Gleichgewicht zwischen Daten- und Physik-Loss zu steuern.
- **Randbedingungen**: Verwenden Sie eine Maske, um Hindernisse im Bild zu identifizieren.

---

### **6. Zusammenfassung**
- **Physikalischer Loss**: Bestraft Abweichungen von der Wellengleichung und Randbedingungen.
- **Implementierung**: Verwenden Sie finite Differenzen oder automatische Differentiation, um Ableitungen zu berechnen.
- **Training**: Kombinieren Sie Daten-Loss und physikalischen Loss, um physikalisch plausible Bilder zu erzeugen.

Falls Sie weitere Fragen zur Implementierung oder zu spezifischen Details haben, lassen Sie es mich wissen!
```





