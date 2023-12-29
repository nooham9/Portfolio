"""
Ciara Maher and Nooha Mohammed
CSE 163

Tests the functions made in analysis and contains the main method.
"""
import pandas as pd
import geopandas as gpd

import load_data as load
import analysis as a


def test_pop_by_state(stressors_and_states) -> pd.DataFrame:
    """
    This testing function is used to make sure that the plot
    is displaying the expected results from percent loss. For
    each year, the listed state should have the darkest color on
    the plotted map. It accepts and returns a pandas DataFrame
    """
    test_data = stressors_and_states[['Year (No quarter)',
                                      'State', 'Percent Loss']]
    test_data = test_data.groupby(['Year (No quarter)',
                                   'State']).max()
    results = (test_data
               .groupby(['Year (No quarter)']).idxmax()['Percent Loss'])
    return results


def test_threats_over_time(finished_stressor: pd.DataFrame):
    """
    This testing function is used to make sure that the plot
    is displaying the expected results of the largest stressors.
    The listed maximum stressor should have the
    largest section in each year in the stacked bar plot.
    """
    stats = (finished_stressor
             .groupby(['Year (No quarter)'])
             [['Varroa Mites (Thousand)',
               'Other pests and parasites (Thousand)',
               'Diseases (Thousand)',
               'Pesticides (Thousand)', 'Other (Thousand)',
               'Unknown (Thousand)']].sum())

    return stats.idxmax(axis=1)


def main():
    """
    Runs all desired functions to be implemented in this program,
    as well as reads in our csv and geopandas data.
    """
    states_df = (
        gpd.read_file(
            r'USA_States_(Generalized)/USA_States_Generalized.shp'
        )
    )
    honey_data = pd.read_csv(r"NASS_Bee-Honey_2015-2021.csv")
    bee_data = pd.read_csv(r"NASS_Bee-Colony_2015-2021.csv")
    stressor_data = pd.read_csv(r"NASS_Bee-Stressors_2015-2021.csv")
    stressor_data = load.clean_stressors(stressor_data)
    bee_data = load.clean_bee(bee_data)
    honey_data = load.clean_honey(honey_data)
    # Use finished_honey for q4
    finished_honey = load.merge_honey_data(honey_data)
    # Use finished_stressor for q1, q2, and q3
    finished_stressor = load.merge_stressor_data(stressor_data, bee_data)
    # Use stressors_and_states for q1
    stressors_and_states = (
        load.merge_stressors_grouped_geometry(finished_stressor, states_df))
    test_pop_by_state(stressors_and_states)
    test_threats_over_time(finished_stressor)

    a.threats_over_time(finished_stressor)
    a.price_stressors(finished_honey, finished_stressor)
    a.pop_by_state(stressors_and_states)
    a.bee_predict(finished_stressor)


if __name__ == '__main__':
    main()
