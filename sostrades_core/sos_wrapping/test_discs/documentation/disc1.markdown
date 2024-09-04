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


#### 2.2 Simplified WBGT Formula

Given that we only have average temperature and relative humidity, we use a simplified formula. This simplification is necessary when complete WBGT 
measurements are not available, which is often the case in large-scale studies or when working with general climate data.

The simplified WBGT formula we use is based on an approximation developed by the Australian Bureau of Meteorology [^1]:

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

Workability is calculated using a two-parameter logistic function [^2]:

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
Research by Zhao et al. (2017) cited in the climate change and future of agri-food production[^3]  indicates that for a 1°C increase in temperature:

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


## Source

[^1]: Applicability of the model presented by Australian Bureau of Meteorology to determine WBGT in outdoor workplaces: A case study,Urban Climate,
Volume 32,2020,100609,ISSN 2212-0955,https://doi.org/10.1016/j.uclim.2020.100609. (https://www.sciencedirect.com/science/article/pii/S2212095519302469)

[^2]: García-León, D., Casanueva, A., Standardi, G. et al. Current and projected regional economic impacts of heatwaves in Europe. Nat Commun 12, 5807 (2021). https://doi.org/10.1038/s41467-021-26050-z

[^3]: Kumar, L., Chhogyel, N., Gopalakrishnan, T., Hasan, M. K., Jayasinghe, S. L., Kariyawasam, C. S., Kogo, B. K., & Ratnayake, S. 
(2023). Chapter 4 - Climate change and future of agri-food production. In Climate Change and Agriculture (pp. 89-110). Academic Press. 
https://www.sciencedirect.com/science/article/pii/B9780323910019000098