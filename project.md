## My Project

I applied machine learning techniques to investigate global particle size distributions (PSDs). Briefly, I reconstructed the global PSDs data by applying a bagged Random Forest (RF) algorithm to a global data set of UVP5 (Underwater Vision Profiler) observations.

Below is my report.

***

## Introduction 

(Here is a summary description of the topic. Here is the problem. This is why the problem is important.)

Export of sinking particles from the surface ocean is critical for carbon sequestration and to provide energy to the deep biosphere. The magnitude and spatial patterns of this export have been estimated in the past by in situ particle flux observations, satellite-based algorithms, and ocean biogeochemical models; however, these estimates remain uncertain.

Here, I present an analysis of particle size distributions (PSDs) from a global compilation of in situ Underwater Vision Profiler 5 (UVP5) optical measurements. Using a machine learning algorithm, I extrapolate sparse UVP5 observations to the global ocean from well-sampled oceanographic variables. I reconstruct global maps of PSD parameters (biovolume [BV] and slope) for particles at the base of the euphotic zone. These reconstructions reveal consistent global patterns, with high chlorophyll regions generally characterized by high particle BV and flatter PSD slope, that is, a high relative abundance of large versus small particles.



(There is some dataset that we can use to help solve this problem. This allows a machine learning approach. This is how I will solve the problem using supervised/unsupervised/reinforcement/etc. machine learning.)

(We did this to solve the problem. We concluded that...)




The Random Forest (RF) algorithm is an example of supervised learning that employs labeled data to teach how to categorize unlabeled data.





I use a bagged RF algorithm to extrapolate the gridded PSD BV and slope at both the euphotic zone and mixed layer depth horizons, based on monthly climatological predictors that include temperature, salinity, silicate, depth, shortwave radiation and other biogeochemical variables such as oxygen, nutrients, chlorophyll, mixed layer depth, net primary production (NPP), euphotic depth, and iron deposition. These predictors can make sure that the model captures both static and denamic aspects of PSD, improving the robustness adn accuracy of the resulting reconstructions.


## Data

(Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!)


<table>
  <tr>
    <th>Feature Category</th>
    <th>Feature</th>
    <th>Dimensions</th>
    <th>Sources</th>
  </tr>
  <tr>
    <td>Universal</td>
    <td>Temperature</td>
    <td>12 $\times$ 102 $\times$ 180 $\times$ 360 </td>
    <td>WOA 18</td>
  </tr>
  <tr>
    <td> </td>
    <td>Salinity</td>
    <td>12 $\times$ 102 $\times$ 180 $\times$ 360 </td>
    <td>WOA 18</td>
  </tr>
  <tr>
    <td> </td>
    <td>Silicate</td>
    <td>12 $\times$ 102 $\times$ 180 $\times$ 360 </td>
    <td>WOA 18</td>
  </tr>
  <tr>
    <td> </td>
    <td>Depth</td>
    <td>12 $\times$ 102 $\times$ 180 $\times$ 360 </td>
    <td>WOA 18</td>
  </tr>
  <tr>
    <td> </td>
    <td>Shortwave Radiation</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>ERA 5</td>
  </tr>
  <tr>
    <td>Oxygen</td>
    <td>Oxygen</td>
    <td>12 $\times$ 102 $\times$ 180 $\times$ 360 </td>
    <td>WOA 18</td>
  </tr>
  <tr>
    <td> </td>
    <td>AOU</td>
    <td>12 $\times$ 102 $\times$ 180 $\times$ 360 </td>
    <td>WOA 18</td>
  </tr>
  <tr>
    <td>Nutrients</td>
    <td>Nitrate</td>
    <td>12 $\times$ 102 $\times$ 180 $\times$ 360 </td>
    <td>WOA 18</td>
  </tr>
  <tr>
    <td> </td>
    <td>Phosphate</td>
    <td>12 $\times$ 102 $\times$ 180 $\times$ 360 </td>
    <td>WOA 18</td>
  </tr>
  <tr>
    <td>CHL</td>
    <td>Merged Chlorophyll</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>GlobColour</td>
  </tr>
  <tr>
    <td> </td>
    <td>Modis Chlorophyll</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>NASA G.S.F.C.</td>
  </tr>
  <tr>
    <td>MLD</td>
    <td>Mixed Layer Depth I</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>MIMOC</td>
  </tr>
  <tr>
    <td> </td>
    <td>Mixed Layer Depth II</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td> de Boyer Montegut et al. (2004) </td>
  </tr>
  <tr>
    <td>NPP</td>
    <td>Eppley VGPM</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>Antoine and Morel (1996) </td>
  </tr>
  <tr>
    <td> </td>
    <td>VGPM</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>Behrenfeld and Falkowski (1997) </td>
  </tr>
  <tr>
    <td> </td>
    <td>CBPM</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>Westberry et al. (2008) </td>
  </tr>
  <tr>
    <td> </td>
    <td>CAFE</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>Silsbe et al. (2016) </td>
  </tr>
  <tr>
    <td>Euphotic Depth </td>
    <td>Eppley VGPM Euphotic depth</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>Morel et al. (2007) </td>
  </tr>
  <tr>
    <td> </td>
    <td>VGPM Euphotic depth</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>Morel et al. (2007) </td>
  </tr>
  <tr>
    <td>  </td>
    <td>CBPM Euphotic depth</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>Morel et al. (2007) </td>
  </tr>
  <tr>
    <td>Iron Deposition </td>
    <td>ELabile Fraction</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>Myriokefalitakis et al. (2018) </td>
  </tr>
  <tr>
    <td> </td>
    <td>Soluble Fraction</td>
    <td>12 $\times$ 180 $\times$ 360 </td>
    <td>Hamilton et al. (2019) </td>
  </tr>
</table>
$


![](assets/IMG/plot1.png)

![](assets/IMG/plot2.png)

(*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*)

## Modelling

(Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. )

(The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.)

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

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

