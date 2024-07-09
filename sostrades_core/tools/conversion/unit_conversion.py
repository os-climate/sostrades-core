'''
Copyright 2022 Airbus SAS
Modifications on 2024/06/28 Copyright 2024 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''
from cmath import pi


class UnitConversion:

    # conversion unit constant
    nm_to_km = 1.85  # nautical mile to kilometer
    m_to_ft = 3.28  # meter to feet
    lb_to_kg = 0.45  # pounds to kilograms
    b_to_hpa = 1000  # bar to hectopascal
    ft_per_min_to_m_per_s = 0.00508  # feet per minute to meter per second
    m_per_s_to_km_per_h = 3.6  # meter per second to kilometer per hour
    deg_to_rad = pi / 180
    kts_to_ms = m_per_s_to_km_per_h / nm_to_km
    gravity = 9.81
