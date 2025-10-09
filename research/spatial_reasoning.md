# Spatial Reasoning with Denoising Models



<a href="../README.md" style="float:right">Back</a>

[Link](https://arxiv.org/abs/2502.21075)



Contents:

- [Explanation](#explanation)
- [Translation from Sudoku to Noise-Prediction](#translation-from-sudoku-to-noise-prediction)
- [Possible Sequentialization](#possible-sequentialization)
- [Open Questions](#open-questions)



<br><br>



---

### Explanation

The Spatial Reason Model (SRM) does not generate the whole output. It masks the image and generates part after part and chooses the next part to generate with a predicted uncertainty. So it makes a sequential generation process with the Chain-of-Thought-Principle in mind.

The gif [from the official GitHub Repo](https://github.com/Chrixtar/SRM) explains it pretty good:

<img src="https://private-user-images.githubusercontent.com/38473808/417544905-23e0538d-8243-4aad-8041-524ff188a732.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTk5NDc1MTAsIm5iZiI6MTc1OTk0NzIxMCwicGF0aCI6Ii8zODQ3MzgwOC80MTc1NDQ5MDUtMjNlMDUzOGQtODI0My00YWFkLTgwNDEtNTI0ZmYxODhhNzMyLmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDA4VDE4MTMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWE5ZmM4MmYyZmRjYTgxM2IxMDI3YWNkOGZhNjljZmNiMmE3OGJmNmM0ZWU1Njc3MmVjOWQxYzhlNmQwNjVlOTcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.x-xp0Q3UcAzY_Oc28FQEcYz4mCZpsysIQLQ-3GlqQ5c"></img>

Dataset have to provide:
-


<br><br>



---

### Translation from Sudoku to Noise-Prediction



| Aspects      | MNIST-Sudoku (Paper)             | Sound-Propagation                           |
| ------------ | -------------------------------- | ------------------------------------------- |
| **Input**    | Partwise Filled Sudoku           | Buildings (Satellite Gray-Image)            |
| **Output**   | Fully Solved Sudoku              | Sound Propagation                           |
| **Rules**    | Sudoku-Logic                     | Physical Laws (Reflexion, Diffraction)      |
| **Problem**  | Rule-break through Hallucination | Physical-Violation through Hallucination    |
| **Solution** | Sequential Generation            | Sequential "Physic-Simulation" / Generation |






<br><br>

---

### Possible Sequentialization

**Sequential Idea**<br>Predict/Generate *from the center to the borders* and *from areas with less buildings to areas with more buildings*.

Or only:

Predict/Generate in radial/circle as the sound waves would propagate.



<br><br>



**End-to-End Way**

1. Repeat:
    1. Predict Uncertainty
    2. Update Mask
    3. Generate



<br><br>



**Residual Way** 

1. Repeat for only `base propagtion`:
    1. Predict Uncertainty
    2. Update Mask
    3. Generate
2. Repeat for only `reflexion`:  (satellite image or result from last step as input)
    1. Predict Uncertainty
    2. Update Mask
    3. Generate
3. Repeat for only `diffraction`: (satellite image or result from last step as input)
    1. Predict Uncertainty
    2. Update Mask
    3. Generate
4. Combine Results (if all only satelite image as input)



> Potential: Model learns first predicting "easy" parts and then focuses on difficult parts



<br><br>

---

### Open Questions



- Is the translation from Sudokus **Logical Rules to Physic Laws**  possible?
- Setup requirements



<br><br>



---









