'''
Copyright 2022 Airbus SAS

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
from gemseo.mda.mda_factory import MDAFactory

def create(
    cls,
    mda_name,  # type: str
    disciplines,  # type: Sequence[MDODiscipline]
    **options  # type: MDAOptionType
):  # type: (...) -> MDA
    """Create a MDA.
    Replace GEMSEO classes by SoSTrades MDA classes

    Args:
        mda_name: The name of the MDA (its class name).
        disciplines: The disciplines.
        **options: The options of the MDA.
    """
    cls.SOS_MDA_SUFFIX = "SoS"
    sos_mda_name = cls.SOS_MDA_SUFFIX + mda_name
    # Checks if the class in its SoS version
    if cls.is_available(sos_mda_name):
        mda_name = sos_mda_name
    return cls.factory.create(mda_name, disciplines=disciplines, **options)
    
    
#setattr(MDAFactory, "create", create)