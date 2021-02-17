import pandas as pd

from .data import load_data

def Populations():
    """Closure, enabling O(1) lookup, using population data from both the age and testing datasets

    Returns:
        callable: Function that returns the population given the name of the country
    """
    age_df = load_data('agerangenotificationeu')[['country', 'population']]
    testing_df = load_data('testing')[['country', 'population']]
    pop = pd.merge(age_df, testing_df, how='outer').groupby(['country']).median()
    
    def getter(country: str) -> float:
        return pop.loc[country, 'population']

    return getter
