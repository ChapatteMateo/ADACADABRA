---
layout: home
title: Impact of wealth on eating habits in Greater London
subtitle: Technical extension 
cover-img: /assets/img/baking_background.jpg
---


## Introduction:

Our story starts from the paper "Tesco Grocery 1.0, a large-scale dataset of grocery purchases in London". It presents a record of 420 millions food items purchased by 1.6 million fidelity card owners who shopped at the 411 Tesco stores in Greater London over the course of the entire year of 2015. The data is aggregated at multiple levels of census areas (borough, ward, MSOA, LSOA) to preserve anonymity.

In the paper, the authors describe the derivations of mutiple food characteritics (associated to each area), mainly concerning the energy, weight, type and entropy of the bought products. The authors then establish the validity of the dataset by comparing food purchase volumes to population from official census and match nutrient and energy intake to official statistics of food-related illnesses. To find out more about the Tesco paper and its characteristics, do not hesitate to consult their [official page](https://springernature.figshare.com/articles/Metadata_record_for_Tesco_Grocery_1_0_a_large-scale_dataset_of_grocery_purchases_in_London/11799765).

Their dataset contains precious information concerning eating habits, indeed it is one of the only studies with such a big scale that contains both geo-location and nutritional information. This raised our interest and made us wonder:

Have you ever heard sentences like _"Healthy eating is a privilege of the rich"_? Well, with the Tesco dataset we have the chance to bring further checkings to this kind of claim and bring an element of answer to the more general assumption **"Do wealthier populations buy healthier food?"**

In order to answer our interrogation, we came up with the following questions:
 - At MSOA level in Greater London, are there major differences in eating habits depending on the wealth level?
 - Is there evidence of social class difference in eating habits?
 - Do wealthier MSOA areas buy food that could be judged healthier?
 
As a side note and disclaimer, let us underline that the following analysis is done as part of the Aplied Data Aanalysis course (EPFL) and that we whish to make this data story on a lighter tone than the core work. If you want to find all the serious explanations and details on the calculations, we advise you to read the [corresponding notebook](https://github.com/ChapatteMateo/ADACADABRA/blob/master/P4-technical_ext_work/extension.ipynb).
 
 So let's get into work and start digging!
 
## Let's get some data:

Let's make a quick (boring but necessary) detour on the different datasets we required and the reason we discarded some MSOA areas in our study.

Concerning the datasets, the one from [Tesco Grocery](https://springernature.figshare.com/articles/Metadata_record_for_Tesco_Grocery_1_0_a_large-scale_dataset_of_grocery_purchases_in_London/11799765) will the basis of our analysis. Additionally, to come up with a wealth estimation for each MSOA, we can rely on the data issued by [Greater London Authority](https://data.london.gov.uk/dataset/msoa-atlas). Indeed they share a wide range of statistical information on MSOA's population, including the median income that we will use as a base for our wealth estimate.

If you've skipped through this section you might wonder _"But why are there some grey areas on the London map ?"_
Well, to increase the significance of the study, we discarded information coming from MSOAs for which the the ratio of people having a clubcard at Tesco among the total population of the area was not representative. For that purpose we used a metric called normalized representativeness. The normalized represenativeness of a given MOSA can be computed as folows:

$$represenativeness_{normalized}=\frac{represenativeness-min(represenativeness)}{max(represenativeness)-min(represenativeness)}$$

where the represenativeness of the given MSOA is:

$$represenativeness=\frac{number \: of \: customers}{population}$$

We followed the same procedure as the original Tesco paper by discarding all MSOAs whose $represenativeness_{normalized}$ was below $0,1$. This procedure leads to the removal of a little less than $10\%$ of the MOSAs.

[si on arrive a faire un representation carte qui donne the nom du MSOA on hover, on la mettrait là en disant, voilà les MSOA restant, c'est joli wow]

<object data="assets/img/figure_map.html">
    Your browser doesn’t support the object tag. 
</object>


### Discretisation into wealth class

We used the median income by area as the main indicator of wealth. The median has the advantage of being robust to strong outliers (which is often the case with income because they are said to follow a Pareto distribution, the 80-20 rule <-ça peut être retiré, comme vous voulez).
We ran PCA on the income and ploted the inertia depending on the number of cluster to find an appropriate number of cluster.
[image du plot d'inertie ? même chose là, on peut tirer le plot avec plot.ly pour a voir une visu interactive]
We choose 5 clusters and labeled the incomes as {very_low','low','medium','high','very_high'}.
[visu des statistique par cluster ?]

## Data, data, show me which population class eats healthy !

Ok data it's time to talk. Let's consider for each MSOAs, the set of nutrient weight as a vector. We will use TSNE visualization to compare eating habits difference between each MSOAs using these vectors. We will do the same with distribution of purchased product types. Let us see what it reveals:

![png](/assets/img/tsne.png)

Haha !! It seems they are no great clusters. But it seems data is trying to hide something. What we can observe is that there is a clear cluster for the very high class and is not overlapping with the very low class in both visulizations. Sorry data, but you said enough so we continue the examination.

For example, show a map of London with both wealth classes and mean weigh of comsumed fibre in each MSOAs. As fibre is good for healh, maybe we'll get an insight.

![png](/assets/img/graphs/choro_map_fibre.png)

Interesting, we might dicern fibers somehow correlates with the wealth class. We might especially remark this lighter diagonal '>' shape (on the right part of the map) that links the very low incomes with low fraction of fibers in the population's diet. But it's quite hard to evaluate how big and significant is the correlation.

Now, we want to know more. Let us increase the temperature so that data shows us beautiful red shades. For each nutrient weight and product type weight, We want to visualize the correlation between the mean weight within each class and the wealth classes.

![png](/assets/img/product_heatmap.png)
![png](/assets/img/nutrient_heatmap.png)

Amazing data !! Thank you for these informations. So it seems that the comsumption of wine, fish, dairy products, fruit&vegetables and beer are correlated with high social class value. Whereas the comsumption of soft drinks, spirits, grains, poultry and sweets are correlated with low social class value. 
Additionally we observe the entropy does not show wealth correlation. We aslo observe that fibre, protein, alcohol and the nutrients entropy are correlated with high wealth class wheras salt, fat, carb and sugar are correlated with low class value. Finally for saturate fat, we don't observe clear correlation.

It seems that all features which are positive correlated are markers of healthy eating wheras the features which are negative correlated are markers of unhealthy eating. But under interrogation, data cans say anything we want it to say. Let's make a last visulization before looking for proofs. 
Data ! Show us the ditribution of nutrients for each wealth class !

![png](/assets/img/PieChart.png)

Well, well, well, actually the differences in proportions are quite small. But if we observe carefully, we see the percentage of protein increases by 0.3-0.6% for each class increase. The same phenomena can be observed for fat, alcohol and fiber. Thus the fraction of carb decreases with higher class level. So it seems there are correlations but there are no great differences in eating habits. That's why we are here, we are looking for the details.

### Section to be removed on the final version -- Markdown help

#### you can write python code like this:

```python
import numpy as np
print("Random stuff")
```

#### you can make a table like this:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area_id</th>
      <th>weight</th>
      <th>...</th>
      <th>man_day</th>
      <th>population</th>
      <th>area_sq_km</th>
      <th>people_per_sq_km</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E02000001</td>
      <td>323.180804</td>
      <td>...</td>
      <td>103934</td>
      <td>6687.0</td>
      <td>2.90</td>
      <td>2305.862069</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E02000002</td>
      <td>397.651232</td>
      <td>...</td>
      <td>9952</td>
      <td>7379.0</td>
      <td>2.16</td>
      <td>3416.203704</td>
    </tr>
  </tbody>
</table>
</div>


#### you can add an image like this:
Note that we stored all images inside the folder assets/img/graphs

![png](/assets/img/graphs/output_14_1.png)

to export plt figure in high definition -> fig.savefig("choro_map.png", bbox_inches='tight', dpi=600) 


#### you can write $\LaTeX$ like this:

$f_{nutrient_i}(a) = \frac{nutrient_i(a)}{\sum_j nutrient_j(a)}$

#### various thingy:

_italic_

**bold**

[link display name](https://real_link_url.ch)

> "quotes"

* elem1
	* sub-elem1_1
	* sub-elem1_2
* elem2
* elem3

1. first
2. second
3. third

