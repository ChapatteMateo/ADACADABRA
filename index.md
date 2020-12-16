---
layout: home
title: Impact of wealth on eating habits in Greater London
subtitle: Technical extension 
cover-img: /assets/img/baking_background.jpg
---


## Introduction:

Our story starts from the paper "Tesco Grocery 1.0, a large-scale dataset of grocery purchases in London". It presents a record of 420 millions food items purchased by 1.6 million fidelity card owners who shopped at the 411 Tesco stores in Greater London over the course of the entire year of 2015. The data is aggregated at multiple levels of census areas (borough, ward, MSOA, LSOA) to preserve anonymity.

In the paper, the authors describe the derivations of mutiple food characteritics (associated to each area), mainly concerning the energy, weight, type and entropy of the bought products. The authors then establish the validity of the dataset by comparing food purchase volumes to population from official census and match nutrient and energy intake to official statistics of food-related illnesses. To find out more about the Tesco paper and its characteristics, please consult their [official page](https://springernature.figshare.com/articles/Metadata_record_for_Tesco_Grocery_1_0_a_large-scale_dataset_of_grocery_purchases_in_London/11799765).

This dataset contains precious information concerning eating habits as it is one of the only studies with such a big scale, containing both geo-location and nutritional information.

This raised our interest and made us wonder. Have you ever heard sentences like _"Healthy eating is a privilege of the rich"_? Well with this Tesco dataset we had the chance to bring further checkings to this kind of claim and check the more general assumption **"Do wealthier populations buy healthier food?"**.

In order to answer this general interrogation, we came up with the following more precise questions:
 - At MSOA level in Greater London, are there major differences in eating habits depending on the wealth level?
 - Is there evidence of social class difference in eating habits?
 - Do wealthier MSOA areas buy food that could be judged healthier?
 
 So let's get into work and start digging!
 
## The data we need:
 
To come up with a wealth estimation for each MSOA, we can rely on the data issued by [Greater London Authority](https://data.london.gov.uk/dataset/msoa-atlas). Indeed they share a wide range of statistial informations on MSOA's population, including the median income that we will use as a base for our wealth estimate.






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

