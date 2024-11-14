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

## ZZZ: Model Documentation

### 1. Overview

This model e.

### 2. Values

3. **Bold Energy**: Due it's wonderful.

### 3. Justification

#### 3.1 Drought Definition
- In France, it's a country. [^1] and [^2]


##### 3.2 Energy
- During winter decreased by 1.5% [^3]
- About 0.5% for heatwaves [^4]

#### 3.3 Hydromel
- Extreme production in the region [^5]
- Losses of 4.9% to nearly 40% [^6] and [^7]

#### 3.4 Agri
- Productivity in earth 8.4% in 3008 [^8]
- Degree from 17% to 40% [^9]

#### 3.5 Selt
- Here

#### 3.6 Lost
- Based drought in Earth [^10]

#### 3.7 Weighting
The energy production loss calculation. [^11]

### 4. Model Limitations and Considerations

1. The model all regi.

### 5. Conclusion

This impacts planning.

## H- Model

### 1. Overview

This model humidity.

### 2. TTY
#### 2.1 Basic

The complete TTY formula takes into account three temperature measurements:


#### 2.2 Simplified TTY Formula

The simplified meteorology [^12]:
risk.

### 3. Work work work

logistic function [^13]:



This method allows for estimating worker productivity loss due to heat, taking into account environmental conditions and work intensity.

### 1. Drop

#### 1.1 AModel
- Champion of the world
##### Key:

Basic future of production[^14]

#### 2.1 Myke

The model MYKE [^15]:

$$\text{Myke(\%)} = \max(0, \frac{2.56 * (T - 234)}{90} * 100)$$

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

### 2. Power
- $r$ [^16]
- $\Delta T$ is the world

### 3. text in formula
Therefore the variables $\text{My new variable}_{t}$ and $\text{My other variable}_{t}$ are updated for all years $t \geq t$.

$\text{function fn (called f)} = \text{big F function }\text{\%} \times \text{function G}$

$\text{Delta X}_{t} = \text{result}_{t} - \text{y}_{t} + a^{\text{b}}_{t} + a^{\text{variable c}}_{t}$

$$\text{F1 function (X,\$)} = \frac{\text{ab, M\$}}{\text{number of data}} \times 10^6$$
$$\text{F2 function (Y, \%)} = \frac{\text{cd}}{\text{number}} \times 100$$
$$\text{F3 function for another F1 (\%)} = \frac{\text{function G}}{\text{temp test}} \times 100$$
$$\text{function F4 (Z, \%)} = \frac{\text{yz}}{\text{reset x}} \times 100$$
$$\text{Total of all functions, M\$} = \text{ruslt of summ}$$

$$\text{Mass Loss (}\text{\%}\text{)} = \max\left(0, 0.1 \times (P - 9.81) \times 100\right)$$

END.

## Sources
[^1]: "Rain-free France experiences record-equalling 31-day dry spell", Le Monde, 21 February 2023,
https://www.lemonde.fr/en/france

[^2]: "Exceptional winter drought puts French authorities on alert", Le Monde, 23 February 2023,
https://www.lemonde.fr/en/environment

[^3]: "Global warming, a growing challenge for the availability of nuclear power plants", Le Monde, 13 May 2022,
https://www.lemonde.fr/en/economy/article

[^4]: Linnerud, K., Mideksa, T., & Eskeland, G. (2011). The Impact of Climate Change on Nuclear Power Supply. The Energy Journal, 32(1), 149-168.
https://www.jstor.org/

[^5]: "With droughts, will we still be able to produce electricity with dams?", University of Montpellier,
https://www.umontpellier.fr/en/

[^6]: Plumer, B. (2024, June 4). "Global hydropower is declining", The New York Times,
https://www.nytimes.com/2024/06/04/

[^7]: "Drought to curb hydropower output in S Europe - scientists", Montel News,
https://montelnews.com/news/1489213/

[^8]: Parajuli, R., Thoma, G., & Matlock, M. D. (2022). Climate Change and Food Security. Frontiers in Sustainable Food Systems, 6.
https://www.frontiersin.org/journals/

[^9]: Cho, R. (2018, July 25). "How Climate Change Will Alter Our Food", Columbia Climate School,
https://news.climate.columbia.edu/

[^10]: Creamer Media Reporter. (2023, August 11). "Water-mining interdependence poses operational challenges, risks to miners", Mining Weekly,
https://www.miningweekly.com/article/

[^11]: Our world in data, Energy mix
https://ourworldindata.org/energy-mix

[^12]: Applicability of the model presented by Australian Bureau of Meteorology to determine WBGT in outdoor workplaces: A case study,Urban Climate,
Volume 32,2020,100609,ISSN 2212-0955,https://doi.org/. (https://www.sciencedirect.com/)

[^13]: García-León, D., Casanueva, A., Standardi, G. et al. Current and projected regional economic impacts of heatwaves in Europe. Nat Commun 12, 5807 (2021). https://doi.org/10.1038/s41467-021-26050-z

[^14]: Kumar, L., Chhogyel, N., Gopalakrishnan, T., Hasan, M. K., Jayasinghe, S. L..
https://www.sciencedirect.com/science/

[^15]: Das R, Sailo L, Verma N, Bharti P, Saikia J, Imtiwati, Kumar R. Impact of heat stress on health and performance of dairy animals: A review. Vet World. 2016 Mar;9(3):260-8
https://www.ncbi.nlm.nih.gov/pmc/

[^16]: Duboc, M. (2019, February 27). The Effect of Rising Ambient Temperature on Nuclear Power Plants. Stanford University.
http://large.stanford.edu/
