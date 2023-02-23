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