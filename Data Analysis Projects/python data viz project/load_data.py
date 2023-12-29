"""
Ciara Maher and Nooha Mohammed
CSE 163

Loads in, cleans, and merges datasets to be used in analysis.
"""
import pandas as pd
import numpy as np
import geopandas as gpd


def clean_stressors(stressor: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds a new column to the stressors df that includes
    just the year, rather than the year and quarter. It also removes
    the United States as a state to avoid skewed data
    """
    stressor['Year (No quarter)'] = stressor['Year'].str[0:4]
    index_us = stressor[stressor['State'] == 'United States'].index
    stressor.drop(index_us, inplace=True)
    stressor = stressor.replace('(NA)', np.nan)
    stressor = stressor.replace('(Z)', np.nan)
    return stressor


def clean_bee(bees: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds a new column to the bees df that includes
    just the year, rather than the year and quarter. It also
    removes the United States as a state to avoid skewed data.
    """
    bees['Year (No quarter)'] = bees['Year'].str[0:4]
    index_us = bees[bees['State'] == 'United States'].index
    bees.drop(index_us, inplace=True)
    bees = bees.replace('(NA)', np.nan)
    bees = bees.replace('(Z)', np.nan)
    return bees


def clean_honey(honey: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the honey dataset by removing the U.S.
    as a state to avoid any skewed data.
    """
    index_us = honey[honey['State'] == 'United States'].index
    honey.drop(index_us, inplace=True)
    honey = honey.fillna(0)
    return honey


def merge_bees_data(bees: pd.DataFrame) -> pd.DataFrame:
    """
    This function groups the bees df by year AND state and
    then merges them together. The resulting df contains
    State, Year, Lost Colonies, and Starting Colonies.
    """
    bees_lost_colonies = (bees
                          .groupby(['Year (No quarter)', 'State'],
                                   as_index=False)['Lost Colonies']
                          .sum())
    bees_starting_colonies = (bees
                              .groupby(['Year (No quarter)', 'State'],
                                       as_index=False)['Starting Colonies']
                              .sum())
    bees_joined = pd.merge(bees_lost_colonies,
                           bees_starting_colonies,
                           how='left',
                           on=['Year (No quarter)', 'State'])
    return bees_joined


def merge_stressor_data(stressors: pd.DataFrame, bees: pd.DataFrame) -> \
                        pd.DataFrame:
    """
    This function joins the stressors and bees column, turns the
    percentages for stressors to be in terms of thousand colonies,
    and then groups by year and state.
    """
    bees_all_colonies = bees[['State',
                              'Year (No quarter)',
                              'Starting Colonies',
                              'Lost Colonies']]
    stressors_joined = pd.merge(bees_all_colonies,
                                stressors,
                                how='left',
                                on=['Year (No quarter)', 'State'])
    convert_dict = {'State': str,
                    'Year (No quarter)': int,
                    'Starting Colonies': float,
                    'Lost Colonies': float,
                    'Varroa Mites (Percent)': float,
                    'Other pests and parasites (Percent)': float,
                    'Diseases (percent)': float,
                    'Pesticides (percent)': float,
                    'Other (percent)': float,
                    'Unknown (percent)': float,
                    'Year': object}
    stressors_joined = stressors_joined.astype(convert_dict)
    stressors_joined[["Varroa Mites (Percent)",
                      "Other pests and parasites (Percent)",
                      "Diseases (percent)",
                      "Pesticides (percent)",
                      "Other (percent)",
                      "Unknown (percent)"]].multiply(
                          stressors_joined["Starting Colonies"],
                          axis="index")
    stressors_joined = stressors_joined.rename(
        columns={'Varroa Mites (Percent)':
                 'Varroa Mites (Thousand)',
                 'Other pests and parasites (Percent)':
                 'Other pests and parasites (Thousand)',
                 'Diseases (percent)':
                 'Diseases (Thousand)',
                 'Pesticides (percent)':
                 'Pesticides (Thousand)',
                 'Other (percent)': 'Other (Thousand)',
                 'Unknown (percent)':
                 'Unknown (Thousand)'
                 })
    stressors_joined = stressors_joined.drop(['Year'], axis=1)
    col1 = ['Year (No quarter)', 'State']
    applied = ['Starting Colonies',
               'Lost Colonies',
               'Varroa Mites (Thousand)',
               'Other pests and parasites (Thousand)',
               'Diseases (Thousand)',
               'Pesticides (Thousand)',
               'Other (Thousand)',
               'Unknown (Thousand)']
    stressors_grouped = stressors_joined.groupby(col1)[applied].sum()
    stressors_grouped = stressors_grouped.reset_index()
    stressors_grouped['Percent Loss'] = stressors_grouped['Lost Colonies'] / \
        stressors_grouped['Starting Colonies']
    return stressors_grouped


def merge_honey_data(honey: pd.DataFrame) -> pd.DataFrame:
    """
    Groups honey data by year AND state and sums up or averages everything
    depending on value. Merges datasets together to produce honey_joined,
    which can be used to answer our research questions.
    """
    honey_groupby_data = honey[['State',
                                'Honey producing colonies (thousand)',
                                'Production (1,000 pounds)',
                                'Year']]
    honey_groupby_df = (honey_groupby_data
                        .groupby(['Year', 'State'])
                        [['Honey producing colonies (thousand)',
                          'Production (1,000 pounds)']]
                        .sum().reset_index())
    honey_mean_data = (honey
                       .groupby(['Year', 'State'])
                       [['Yield per colony (pounds)',
                         'Average price per pound (dollars)']]
                       .mean().reset_index())
    honey_joined = pd.merge(honey_groupby_df,
                            honey_mean_data,
                            how='left',
                            on=['Year', 'State'])
    honey_joined['Value of Production (1,000 dollars)'] = \
        honey_joined['Production (1,000 pounds)'] * \
        honey_joined['Average price per pound (dollars)']
    convert_dict = {'Year': int, 'State': str,
                    'Honey producing colonies (thousand)': float,
                    'Production (1,000 pounds)': float,
                    'Yield per colony (pounds)': float,
                    'Average price per pound (dollars)': float,
                    'Value of Production (1,000 dollars)': float}
    honey_joined = honey_joined.astype(convert_dict)
    return honey_joined


def merge_stressors_grouped_geometry(stressors_grouped: pd.DataFrame,
                                     states_df: gpd.GeoDataFrame) -> \
                                         pd.DataFrame:
    states_df = states_df[['STATE_NAME', 'STATE_ABBR']]
    states_df = states_df.rename(columns={'STATE_NAME': 'State',
                                          'STATE_ABBR': 'Code'})
    stressors_bees_states = stressors_grouped.merge(states_df,
                                                    left_on='State',
                                                    right_on='State',
                                                    how='inner')
    return stressors_bees_states
