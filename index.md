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
 
## Let's get DATA:

Our best contact to get some inisghs about our questions is called DATA. He knows everything about:

	- [Tesco Grocery](https://springernature.figshare.com/articles/Metadata_record_for_Tesco_Grocery_1_0_a_large-scale_dataset_of_grocery_purchases_in_London/11799765). Additionally, he come up with a wealth estimation for each MSOA, we can rely on the data issued by [Greater London Authority](https://data.london.gov.uk/dataset/msoa-atlas). Indeed they share a wide range of statistical information on MSOA's population, including the median income that we will use as a base for our wealth estimate.

[If you've skipped through this section you might wonder _"But why are there some grey areas on the London map ?"_]
DATA also discarded information coming from MSOAs for which the the ratio of people having a clubcard at Tesco among the total population of the area was not representative. For that purpose we used a metric called normalized representativeness. The normalized represenativeness of a given MOSA can be computed as folows:

[MOi les formules je les enlèverais mais voir si on peut inclure ça de manière sympas :)]
$$represenativeness_{normalized}=\frac{represenativeness-min(represenativeness)}{max(represenativeness)-min(represenativeness)}$$

where the represenativeness of the given MSOA is:

$$represenativeness=\frac{number \: of \: customers}{population}$$

DATA followed the same procedure as the original Tesco paper by discarding all MSOAs whose $represenativeness_{normalized}$ was below $0,1$. This procedure leads to the removal of a little less than $10\%$ of the MOSAs.

[si on arrive a faire un representation carte qui donne the nom du MSOA on hover, on la mettrait là en disant, voilà les MSOA restant, c'est joli wow]

<object data="assets/img/figure_map.html" width="1000" height="600">
    Your browser doesn’t support the object tag. 
</object>

So DATA will answer our questions for sure but we need to ask the right questions. First We would like to know what are those classes.

### DATA, what are the wealth classes ?

"I used the median income by area as the main indicator of wealth. The median has the advantage of being robust to strong outliers (which is often the case with income because they are said to follow a Pareto distribution.
I ran PCA..."

Blablabla, we don't care about this, just show us some stats ! 

"Ok ok, here are your stats..."

![png](/assets/img/clusters_stat.png)

## DATA, DATA, show me which population class eats healthy !

Ok DATA, from now, I don't want you to talk, just show us the answer !

Let's consider for each MSOAs, the set of nutrient weight as a vector. We will use TSNE visualization to compare eating habits difference between each MSOAs using these vectors. TSNE modelizes distances between vectors, so two MSOAs close to each others have similar vectors, wheras if they are far away to one another the vector are dissimilar. with similar vector are close to each other wheras they are far if vectors are dissimilar.  We will do the same with distribution of purchased product types. Let us see what it reveals:

![png](/assets/img/tsne.png)

Hmmmm, it seems they are no great clusters... but it also seems that data is trying to hide something. What we can observe is that there is a clear cluster for the very high class and is not overlapping with the very low class in both visulizations. Sorry DATA, we will need to continue your examination in more details...

Show us for example a map of London with both wealth classes and mean weight of comsumed fibre in each MSOAs. As we know, fibre is good for health so maybe we'll get an insight !

![png](/assets/img/graphs/choro_map_fibre.png)

Interesting... we dicern fiber somehow correlates with the wealth class. Especially, let's remark this lighter diagonal '>' shape (on the right part of the map) that links the very low incomes with low weight of fibers in the population's diet. However it is quite hard to evaluate how big and significant is the correlation.

Now, we want to know more. Let us increase the temperature so that DATA shows us beautiful red shades. For each nutrient weight and product type weight, we would like to visualize the correlation between the mean weight within each class and the wealth classes.

![png](/assets/img/product_heatmap.png)
![png](/assets/img/nutrient_heatmap.png)

Amazing DATA !! Thank you for these valuable informations. So it seems that the comsumption of wine, fish, dairy products, fruit&vegetables and beer are correlated with high social class value, whereas the comsumption of soft drinks, spirits, grains, poultry and sweets are correlated with low social class value. 
Additionally we observe the entropy does not show wealth correlation. We also observe that fibre, protein, alcohol and the nutrients entropy are correlated with high wealth class wheras salt, fat, carb and sugar are correlated with low class value. Finally for saturate fat, we don't observe clear correlation.

It seems that all features which are positively correlated are markers of healthy eating, wheras the features which are negatively correlated are markers of unhealthy eating. But under interrogation, couldn't DATA always tells us what we want to hear ? Let's make a last visulization before looking for proofs. 
DATA ! Show us the distribution of nutrients for each wealth class !

![png](/assets/img/PieChart.png)

Well, well, well, actually the differences in proportions are quite small. But if we observe carefully, we see the percentage of protein increases by 0.3% up to 0.6% for each class increase. The same phenomena can be observed for fat, alcohol and fiber, thus the fraction of carb decreases with higher class level. So it seems there are correlations. The differences in eating habits are no huge, however as data analysts, we are here especially for these details.


## DATA, it's time for lie detector test

### Correlation test

DATA, we will ask you questions about two set of features. First the weigh distribution of purchased product types and then the mean weight of each nutrient within these products. It's time to pass the spearman correlation test, we will now if you lie using the p_value. It it is below 0.05, then we will reject your sayings. Then let's go with product type first:

![png](/assets/img/corr_product.png)

Well, all above correlations are statistically significant. DATA is not lying to us, fine ! We note strong postitive correlations between wealth class and both fruit&vegetables, dairy, fish which are markers of healthy eating habits. We note strong negative correlation between wealth class and sweets, soft drinks which are marker of unhealty eating habits.

We also note a strong positive correlation for wine and negative correlation for water.
We can only make hypothesis about this strong correlation, the most likely one that we could think of is that the amount of water bought does not vary significanlty but because the wealthier population buys more products then the fraction of them represented by water is reduced. About wine, is it not surprising that, as it is wealthy product, its consumption is correlated with the median income.

Ok DATA, we will make same test on nutrients now:

![png](/assets/img/corr_nutrients.png)

Perfect ! All correlations are statistically significant except for saturate (p_value=0.1), so we reject this correlation. We indeed note strong positive correlation between both fibre, entropy, protein, and alcohol with median income wheras we note negative correlation with fat, carb, sugar and salt. If we consider protein, fibre and entropy as health markers, we indeed have a strong correlation health - wealth. Good DATA, good !!

### Logistic test
Finally, we have a last question to ask. What features will you use to predict the wealth class ? Note that by answering this question we will know whether or not the wealth impacts eating habits, so be clear and concise. [Link to explanation why ?]

The test will be simple, we will group medium, low and very low class into non wealthy class and group high and very high into wealthy class. By cumulating fraction values of healthy products (dairy, fish, fruit&veg) into healthy and cumlating fraction values of unhealthy products (fats oils, sweets) into unhealthy, could you predict the wealth class DATA ?

![png](/assets/img/coeffs_products.png)

Congratulation DATA, you managed to explain 29% of the variance with those features. Plus the coefficient of `unhealthy` and the combined `unhealthy:healthy` are statistically significant having a p_value below 0.05. But we reject `healthy` as not significant (p_value=0.27). The `unhealthy` coefficient is negative meaning buying more unhealthy product lowers probabilty for the area to be wealthy. We note that fractions features are dependent since they sum to one. That's why we combined healthy to unhealthy. This last one has positive coefficient and is statistically significant. Combining this to the fact that `unhealthy` is also significant means that areas where fractions of unhealthy product are going to healthy product have higher probability to be wealthy.

The last test data, let launch the same experiment with nutrients plus the nutrients weight entropy:

![png](/assets/img/coeffs_nutrient.png)

Very good DATA !! Now you managed to explain 45% of the variance, we will keep this version as it is better. All coefficients except salt, protein and carb are statistically significant. Both `h_nutrients_weight`, `fibre`, `sugar` and `saturate` have positive coefficients whereas `fat` has negative coefficients. We conclude the entropy has the highest influence on wealth class with fibre which are markers of health.

Thank you for your cooperation DATA, we'll likely see each other in the next days.


## Conclusions and limitations



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

