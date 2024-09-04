# Documentation test

On rc :

```mermaid
graph LR
    A((RC)) --> B(Propulsion)
    A --> C(Make Airframe/Equipment)
    A --> N(Liability)
    N --> O[Escalation]
    B --> F[Escalation by phases]
    C -- 1- %Labour --> D(Material)
    C -- %Labour --> E(Labour)
    D --> G[Escalation]
    E -- % Common --> K[Escalation and Productivity]
    E -- % Modify --> L[Escalation, Learning Curve and Productivity]
    E -- % New --> M[Escalation, Learning curve and Productivity]
```

$$RC = RC\_Propulsion + RC\_Make + RC\_Liability$$
$$RC\_Make = (1-Perc\_Labour)*RC\_Material + \\
Perc\_Labour*RC\_Labour$$
![Test Image](test_doc_image.PNG)
Percentages on Labour (30%), Common (60%), New (30%) and Modify (10%) can be easily changed.

## Drought Impact on Productivity: Model Documentation

### 1. Overview

This model estimates the productivity loss in various sectors due to drought conditions. It uses a logistic (S-curve) function to model the relationship between the number of consecutive dry days and the percentage of productivity loss.

### 2. Key Components

1. **Sector-specific maximum productivity loss**: Based on historical data and research, we've established maximum productivity loss percentages for each sector during extreme drought conditions.

2. **Logistic function**: We use an S-curve to model the progression of productivity loss as drought conditions worsen.

3. **Energy sector special handling**: Due to the importance and unique characteristics of the energy sector, we model nuclear and hydropower separately and combine them weighted by their contribution to global energy production.

### 3. Justification and Data Sources

#### 3.1 Drought Definition
- In France, 31-32 consecutive days without rain is considered an extreme drought, especially in winter. We use this definition as a generalization. [^1] and [^2]


##### 3.2 Nuclear Energy
- Maximum loss set at 10% (conservative estimate)
- During the 2003 extreme drought, nuclear production decreased by 1.5% [^3]
- Studies show that a 1°C rise in temperature reduces nuclear power supply by about 0.5%, with losses exceeding 2% per degree during droughts and heatwaves [^4]

#### 3.3 Hydropower
- Maximum loss set at 35%
- Extreme drought in 2022 caused a 35% decrease in hydroelectric production in the Rhône region [^5]
- Global studies show extreme drought can cause hydropower production losses of 4.9% to nearly 40% [^6] and [^7]

#### 3.4 Agriculture
- Maximum loss set at 28% (average of reported ranges)
- Drought reduced agricultural productivity in South Africa by 8.4% in 2015 [^8]
- Extreme drought in Asia caused agricultural productivity losses ranging from 17% to 40% [^9]

#### 3.5 Manufacturing
- Maximum loss set at 25% (estimate based on qualitative reports)
- No specific figures available, but sources confirm major impacts during drought conditions

#### 3.6 Mining
- Maximum loss set at 24%
- Based on reported production losses of up to 24% during extreme drought in Chile [^10]

#### 3.7 Energy Sector Weighting
Nuclear (3.27%) and hydropower (6.73%) account for 10% of global primary energy consumption. We use these proportions to weight their contributions in the energy production loss calculation. [^11]

### 4. Model Limitations and Considerations

1. The model assumes a uniform impact across all regions, which may not be accurate for global applications.
2. It doesn't account for adaptation measures or technological improvements that might mitigate drought impacts.
3. The S-curve approach is a simplification of complex, often non-linear relationships between drought duration and productivity loss.
4. The model doesn't consider secondary or cascading effects of drought across sectors.

### 5. Conclusion

This model provides a data-driven approach to estimating productivity losses due to drought across various sectors. While it has limitations, it offers a structured method for assessing potential impacts, which can be valuable for risk assessment and planning purposes.

## Heat-Related Worker Productivity Loss Calculation Model

### 1. Overview

This model calculates the productivity loss of workers due to heat, based on the type of work environment, average temperature, and relative humidity.

### 2. WBGT (Wet Bulb Globe Temperature) Calculation

#### 2.1 Basic WBGT Formula

The complete WBGT formula takes into account three temperature measurements:

$$WBGT = 0.7 * Tw + 0.2 * Tg + 0.1 * Ta$$

Where:
- Tw: Wet bulb temperature
- Tg: Globe temperature
- Ta: Air temperature

#### 2.2 Simplified WBGT Formula

Given that we only have average temperature and relative humidity, we use a simplified formula. This simplification is necessary when complete WBGT 
measurements are not available, which is often the case in large-scale studies or when working with general climate data.

The simplified WBGT formula we use is based on an approximation developed by the Australian Bureau of Meteorology [^12]:

$$WBGT = 0.567 * Ta + 0.393 * e + 3.94$$

Where:
- Ta: Air temperature in °C
- e: Water vapor pressure

The water vapor pressure (e) is calculated as follows:

$$e = (RH / 100) * 6.105 * exp((17.27 * Ta) / (237.7 + Ta))$$

Where:
- RH: Relative humidity in percentage

This simplified formula provides a reasonable estimate of WBGT when only air 
temperature and relative humidity are available, making it useful for many 
practical applications and large-scale assessments of heat stress risk.

### 3. Workability Calculation

Workability is calculated using a two-parameter logistic function [^13]:

$$workability = 0.1 + \frac{0.9}{1 + (WBGT / \alpha_1)^{\alpha_2}}$$

Where α1 and α2 are parameters that vary according to work intensity:

- Light work: α1 = 34.64, α2 = 22.72
- Moderate work: α1 = 32.93, α2 = 17.81
- Heavy work: α1 = 30.94, α2 = 16.64

### 4. Productivity Loss Calculation

Productivity loss is simply the inverse of workability:

$$productivity loss = 1 - workability$$ 

### 5. Complete Process

1. Determine work intensity based on asset type.
2. Calculate WBGT from temperature and relative humidity.
3. Calculate workability based on WBGT and work intensity.
4. Calculate productivity loss.

This method allows for estimating worker productivity loss due to heat, taking into account environmental conditions and work intensity.

## Specific Agricultural Impact Model

This model estimates the impact of temperature increases on agricultural productivity, focusing on crop yields, milk production, and 
meat production.

### 1. Crop Loss
Crop loss is estimated using an established crop model, accounting for 67% of total agricultural impact.

#### 1.1 Agricultural Production Loss Estimation Model
This model estimates the percentage loss in agricultural production based on the increase in average temperature. It is derived from research on the impact of temperature increase on major global crops.

##### Key Assumptions:

The model focuses on four major crops: wheat, rice, maize, and soybean, which collectively account for over 67% of human caloric intake globally.
A linear relationship is assumed between temperature increase and crop yield reduction.
The model provides a generalized estimate and does not account for regional variations or crop-specific adaptations.
Basis:
Research by Zhao et al. (2017) cited in the climate change and future of agri-food production[^14]  indicates that for a 1°C increase in temperature:

Wheat production decreases by approximately 6%
Rice production decreases by approximately 3%
Maize production decreases by approximately 7.4%
Soybean production decreases by approximately 3.1%
Model Equation:
$$L = \min(T * (0.30W + 0.25R + 0.25M + 0.20S), 100)$$

Where:

L = Estimated percentage loss in agricultural production

T = Temperature increase in °C (in the model we consider the average temperature of the first available data as reference)

W = Percentage loss for wheat (6%)

R = Percentage loss for rice (3%)

M = Percentage loss for maize (7.4%)

S = Percentage loss for soybean (3.1%)

The coefficients (0.30, 0.25, 0.25, 0.20) represent the estimated global production share of each crop.

##### Explanation:
The equation calculates a weighted average of the percentage losses for each crop, based on their estimated global production share. This average loss (4.975% per °C) is then multiplied by the temperature increase to estimate the total percentage loss. The result is capped at 100% to prevent unrealistic estimates.

### 2. Livestock Loss

Livestock loss is divided into milk and meat production losses, accounting for 33% of total agricultural impact.

#### 2.1 Milk Production Loss

The model is based on the relationship between Temperature Humidity Index (THI) and milk yield[^15]:

$$\text{Milk Loss (\%)} = \max(0, \frac{0.41 * (T - 21)}{30} * 100)$$

Where:
- T is the temperature in °C
- 21°C is approximated as the THI threshold of 69
- 30 kg is assumed as the average daily milk production per cow

#### 2.2 Meat Production Loss

The model assumes a direct relationship between temperature increase and reduction in dry matter intake[^15]:

$$\text{Meat Loss (\%)} = \max(0, 0.85 * (T - 21) * 100)$$

Where:
- T is the temperature in °C
- 21°C is assumed as the thermoneutral zone for cattle

### 3. Total Agricultural Loss Calculation

The total agricultural loss due to temperature increase is calculated as follows:

$$\text{Total Agricultural Loss} = 0.67 * \text{Crop Loss} + 0.33 * \text{Livestock Loss}$$

Where:

$$\text{Livestock Loss} = 0.2 * \text{Milk Loss} + 0.8 * \text{Meat Loss}$$

This model provides a comprehensive estimate of agricultural losses by combining impacts on crops and livestock, weighted according to 
their respective contributions to overall agricultural production.

## Energy Sector Productivity Loss Model

This model estimates the productivity loss in the energy sector due to temperature increases, focusing primarily on the impact on 
nuclear power plants.

### 1. Temperature Change Calculation

The model first calculates the temperature increase from a base year:

$$\Delta T = T_{current} - T_{base}$$

Where:
- $\Delta T$ is the temperature increase
- $T_{current}$ is the current temperature in °C
- $T_{base}$ is the average temperature of the base year

### 2. Nuclear Power Productivity Loss

The productivity loss for nuclear power plants is calculated based on the temperature increase:

$$L_{nuclear} = \max(0, r * \Delta T)$$

Where:
- $L_{nuclear}$ is the productivity loss for nuclear power plants
- $r$ is the loss rate, set to 0.00545 (average of 0.37% and 0.72% per °C)[^16]
- $\Delta T$ is the temperature increase

### 3. Overall Energy Sector Loss

The overall energy sector loss is calculated by applying the nuclear share to the nuclear power loss:

$$L_{energy} = L_{nuclear} * \frac{S_{nuclear}}{100}$$

Where:
- $L_{energy}$ is the overall energy sector productivity loss
- $L_{nuclear}$ is the productivity loss for nuclear power plants
- $S_{nuclear}$ is the nuclear share in the energy mix (in percentage)

This model provides an estimate of energy sector productivity loss due to temperature increases, focusing on the impact on nuclear 
power plants which are particularly sensitive to temperature changes.

## Combining All Productivity Losses

This model combines different types of productivity losses to calculate an overall productivity loss for each asset.

### 1. Combination Formula

The overall productivity loss is calculated using the following formula:

$$L_{total} = 1 - ((1 - L_{drought}) * (1 - L_{heat\_workers}) * (1 - L_{heat\_specific}))$$

Where:
- $L_{total}$ is the overall productivity loss
- $L_{drought}$ is the productivity loss due to drought
- $L_{heat\_workers}$ is the productivity loss due to heat impact on workers
- $L_{heat\_specific}$ is the productivity loss due to specific heat impacts (e.g., on energy production or agriculture)

### 2. Explanation

This formula combines the different types of productivity losses in a way that:

1. Ensures that if any individual loss is 100% (1), the overall loss will be 100%.
2. If all individual losses are 0%, the overall loss will be 0%.
3. Appropriately accounts for the combined effect of multiple partial losses, avoiding double-counting.

The formula works by calculating the remaining productivity after each type of loss and then multiplying these together. The result is 
then subtracted from 1 to give the total loss.

For example:
- If drought causes a 20% loss, heat impact on workers causes a 30% loss, and specific heat impacts cause a 10% loss:
- Remaining productivity: (1 - 0.2) * (1 - 0.3) * (1 - 0.1) = 0.504
- Total loss: 1 - 0.504 = 0.496 or 49.6%

This approach provides a comprehensive assessment of productivity loss by considering multiple risk factors while ensuring that the 
combined impact is not overestimated.

## Sources
[^1]: "Rain-free France experiences record-equalling 31-day dry spell", Le Monde, 21 February 2023, 
https://www.lemonde.fr/en/france/article/2023/02/21/rain-free-france-experiences-record-equalling-31-day-dry-spell_6016674_7.html

[^2]: "Exceptional winter drought puts French authorities on alert", Le Monde, 23 February 2023, 
https://www.lemonde.fr/en/environment/article/2023/02/23/exceptional-winter-drought-puts-french-authorities-on-alert_6017027_114.html

[^3]: "Global warming, a growing challenge for the availability of nuclear power plants", Le Monde, 13 May 2022, 
https://www.lemonde.fr/en/economy/article/2022/05/13/global-warming-a-growing-challenge-for-the-availability-of-nuclear-power-plants_5983333_19.html

[^4]: Linnerud, K., Mideksa, T., & Eskeland, G. (2011). The Impact of Climate Change on Nuclear Power Supply. The Energy Journal, 32(1), 149-168. 
https://www.jstor.org/stable/41323396

[^5]: "With droughts, will we still be able to produce electricity with dams?", University of Montpellier, 
https://www.umontpellier.fr/en/articles/avec-les-secheresses-pourra-t-on-toujours-produire-de-lelectricite-avec-des-barrages

[^6]: Plumer, B. (2024, June 4). "Global hydropower is declining", The New York Times, 
https://www.nytimes.com/2024/06/04/climate/global-hydropower-decline.html

[^7]: "Drought to curb hydropower output in S Europe - scientists", Montel News, 
https://montelnews.com/news/1489213/drought-to-curb-hydropower-output-in-s-europe--scientists-

[^8]: Parajuli, R., Thoma, G., & Matlock, M. D. (2022). Climate Change and Food Security. Frontiers in Sustainable Food Systems, 6. 
https://www.frontiersin.org/journals/sustainable-food-systems/articles/10.3389/fsufs.2022.838824/full

[^9]: Cho, R. (2018, July 25). "How Climate Change Will Alter Our Food", Columbia Climate School, 
https://news.climate.columbia.edu/2018/07/25/climate-change-food-agriculture/

[^10]: Creamer Media Reporter. (2023, August 11). "Water-mining interdependence poses operational challenges, risks to miners", Mining Weekly, 
https://www.miningweekly.com/article/water-mining-interdependence-poses-operational-challenges-risks-to-miners-2023-08-11

[^11]: Our world in data, Energy mix
https://ourworldindata.org/energy-mix

[^12]: Applicability of the model presented by Australian Bureau of Meteorology to determine WBGT in outdoor workplaces: A case study,Urban Climate,
Volume 32,2020,100609,ISSN 2212-0955,https://doi.org/10.1016/j.uclim.2020.100609. (https://www.sciencedirect.com/science/article/pii/S2212095519302469)

[^13]: García-León, D., Casanueva, A., Standardi, G. et al. Current and projected regional economic impacts of heatwaves in Europe. Nat Commun 12, 5807 (2021). https://doi.org/10.1038/s41467-021-26050-z

[^14]: Kumar, L., Chhogyel, N., Gopalakrishnan, T., Hasan, M. K., Jayasinghe, S. L., Kariyawasam, C. S., Kogo, B. K., & Ratnayake, S. 
(2023). Chapter 4 - Climate change and future of agri-food production. In Climate Change and Agriculture (pp. 89-110). Academic Press. 
https://www.sciencedirect.com/science/article/pii/B9780323910019000098

[^15]: Das R, Sailo L, Verma N, Bharti P, Saikia J, Imtiwati, Kumar R. Impact of heat stress on health and performance of dairy animals: A review. Vet World. 2016 Mar;9(3):260-8
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4823286/

[^16]: Duboc, M. (2019, February 27). The Effect of Rising Ambient Temperature on Nuclear Power Plants. Stanford University. 
http://large.stanford.edu/courses/2018/ph241/duboc1/