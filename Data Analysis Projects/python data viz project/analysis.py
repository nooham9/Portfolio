"""
Ciara Maher and Nooha Mohammed
CSE 163

Writes functions to analyze bee stressor and honey data.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def pop_by_state(stressors_and_states: pd.DataFrame) -> \
                 plotly.graph_objects.Figure:
    """
    Takes a pandas dataframe and creates a map plot 
    answering research question 1:
    Looking at multiple diseases and threats, in which state(s)
    has the honey bee population been hit hardest since 2015?
    This function takes in a pandas dataframe and displays a
    plot.
    """
    fig = px.choropleth(stressors_and_states,
                        locations='Code',
                        color='Percent Loss',
                        animation_frame='Year (No quarter)',
                        locationmode='USA-states',
                        scope="usa",
                        range_color=[0,
                                     stressors_and_states['Percent Loss'].max()
                                     ],
                        color_continuous_scale='YlOrRd'
                        )
    fig.update_layout(title_text='Lost Bee Colonies Percent in U.S. States')
    fig.show()


def threats_over_time(finished_stressor: pd.DataFrame) -> \
                      plotly.graph_objects.Figure:
    '''
    Takes a pandas dataframe to create a bar chart answering research
    question 2: How did the greatest threats to honey bee populations
    change over 2015-2021.
    '''
    bar_chart = px.bar(finished_stressor,
                       x="Year (No quarter)",
                       y=['Varroa Mites (Thousand)',
                          'Other pests and parasites (Thousand)',
                          'Diseases (Thousand)',
                          'Pesticides (Thousand)',
                          'Other (Thousand)',
                          'Unknown (Thousand)'], title="Stressor")
    bar_chart.show()


def price_stressors(finished_honey: pd.DataFrame,
                    finished_stressor: pd.DataFrame) -> sns.FacetGrid:
    """
    Takes two pandas dataframes and creates 3 plots to answer research
    question 4: How has the price of honey fluctuated over 2015-2021?
    """
    price_group = (finished_honey
                   .groupby(['Year'])
                   [['Average price per pound (dollars)']]
                   .mean())

    sns.relplot(data=price_group,
                x='Year',
                y='Average price per pound (dollars)',
                kind='line')
    plt.savefig('plot1.png')

    production_group = (finished_honey
                        .groupby(['Year'])
                        [['Value of Production (1,000 dollars)']]
                        .sum())
    sns.relplot(data=production_group,
                x="Year",
                y="Value of Production (1,000 dollars)",
                kind='line')
    plt.savefig('plot2.png')

    lost_colonies_group = (finished_stressor
                           .groupby(['Year (No quarter)'])
                           [['Lost Colonies']]
                           .sum())
    sns.relplot(data=lost_colonies_group,
                x="Year (No quarter)",
                y="Lost Colonies",
                kind='line')
    plt.savefig('plot3.png')


def bee_predict(data) -> object:
    '''
    Takes a pandas dataframe and uses an ML model to answer research
    question 3: Looking to the future, how can we predict the honey
    bee populations to change in 5 years?
    '''
    train_size = 0.8

    data = data.dropna()
    features = data.drop(['Lost Colonies',
                          'Year (No quarter)',
                          'State',
                          'Percent Loss',
                          'Starting Colonies'], axis=1)
    labels = data['Percent Loss']
    # Split the data into training, validation, and test sets
    X_train, X_rem, y_train, y_rem = train_test_split(features,
                                                      labels,
                                                      train_size=train_size,
                                                      random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem,
                                                        y_rem,
                                                        test_size=0.5,
                                                        random_state=42)
    # Fit a random forest regressor model on the training data
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    # Make predictions on the test set and compute the mean squared error
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean squared error on test set: {mse:.2f}")
    # Visualize the predicted versus actual values on the test set
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)],
             [min(y_test), max(y_test)],
             '--', color='red')
    plt.xlabel("Actual Percent Loss")
    plt.ylabel("Predicted Percent Loss")
    plt.title("Test Set Performance")
    plt.show()
    # Make predictions on the validation set and compute the mean squared error
    y_pred = regressor.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    print(f"Mean squared error on validation set: {mse:.2f}")
    # Visualize the predicted versus actual values on the validation set
    plt.scatter(y_valid, y_pred)
    plt.plot([min(y_valid), max(y_valid)],
             [min(y_valid), max(y_valid)],
             '--', color='red')
    plt.xlabel("Actual Percent Lost")
    plt.ylabel("Predicted Percent Lost")
    plt.title("Validation Set Performance")
    plt.show()
    return regressor
