## My Project

I applied machine learning techniques to investigate global particle size distributions. Below is my report.

***

## Introduction 

(Here is a summary description of the topic. Here is the problem. This is why the problem is important.)

Export of sinking particles from the surface ocean is critical for carbon sequestration and to provide energy to the deep biosphere. The magnitude and spatial patterns of this export have been estimated in the past by in situ particle flux observations, satellite-based algorithms, and ocean biogeochemical models; however, these estimates remain uncertain.

Here, I present an analysis of particle size distributions (PSDs) from a global compilation of in situ Underwater Vision Profiler 5 (UVP5) optical measurements. Using a machine learning algorithm, I extrapolate sparse UVP5 observations to the global ocean from well-sampled oceanographic variables. I reconstruct global maps of PSD parameters (biovolume [BV] and slope) for particles at the base of the euphotic zone. These reconstructions reveal consistent global patterns, with high chlorophyll regions generally characterized by high particle BV and flatter PSD slope, that is, a high relative abundance of large versus small particles.







(There is some dataset that we can use to help solve this problem. This allows a machine learning approach. This is how I will solve the problem using supervised/unsupervised/reinforcement/etc. machine learning.)

(We did this to solve the problem. We concluded that...)

## Data

(Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!)

![](assets/IMG/datapenguin.png){: width="500" }

(*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*)

## Modelling

(Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. )

(The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.)

(```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```)

(This is how the method was developed.)

## Results

(Figure X shows... [description of Figure X].)

## Discussion

(From Figure X, one can see that... [interpretation of Figure X].)

## Conclusion

(Here is a brief summary. From this work, the following conclusions can be made:)
* first conclusion
* second conclusion

(Here is how this work could be developed further in a future project.)

## References
[1] DALL-E 3

[back](./)

