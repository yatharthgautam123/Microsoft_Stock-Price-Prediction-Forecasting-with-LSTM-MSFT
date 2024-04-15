# Microsoft_Stock-Price-Prediction-Forecasting-with-LSTM-MSFT

It is impossible to predict the future price of a stock with perfect accuracy. However, you can analyze historical data to find patterns that might help you make educated guesses about future trends. The code you provided demonstrates a common approach to financial time series forecasting with LSTMs (Long Short-Term Memory) networks. Here's a breakdown of what the code does:

## Importing libraries and data:
Imports pandas (pd) for data manipulation
Imports matplotlib.pyplot (plt) for plotting
Reads the CSV file containing Microsoft stock data (MSFT.csv) into a pandas dataframe (df)
Selects 'Date' and 'Close' columns and creates a new dataframe (df)

## Data Preprocessing:
Converts the 'Date' column to datetime format
Function to convert string date to datetime:
Takes a date string in YYYY-MM-DD format
Splits the string into year, month, day
Returns a datetime object

## Exploratory Data Analysis (EDA):
Plots the closing price of the stock over time (plt.plot(df.index, df['Close']))

### Function to create windowed data for training:
This function takes a dataframe, a start date string, an end date string, and a window size (n) as input.
It iterates over dates between the start and end date.
For each date, it extracts the previous n closing prices from the dataframe.
If there are not enough data points for the window, it prints an error message.
It creates a new dataframe with columns: 'Target Date', 'Target-i' (where i represents the lag from the target date), and 'Target' (the closing price on the target date).
It returns the new dataframe.

### Function to convert windowed data into training format:
Takes a windowed dataframe as input
Converts the dataframe to a numpy array
Separates the 'Target Date' column, the features ('Target-i' columns), and the target variable ('Target' column)
Converts the features and target to float32 data type
Returns the dates, features (reshaped as a 3D array), and target

### Train-Test-Validation Split:

Splits the data into training (80%), validation (10%), and test (10%) sets based on the number of dates

## Plotting the training, validation, and test data:
Plots the closing price for each set to visualize the split

## Building the LSTM Model:
Creates a Sequential model from TensorFlow.keras.models
Adds an LSTM layer with 64 units as the first layer
Adds two Dense layers with 32 units each and a 'relu' activation function
Adds a final Dense layer with 1 unit for the predicted closing price
Compiles the model with Adam optimizer (learning rate=0.001) and mean squared error (mse) loss function with mean absolute error (mae) as a metric

## Training the Model:
Fits the model on the training data with validation data for 100 epochs

### It's important to understand the limitations:
Machine learning models cannot predict the future with certainty.
Historical data may not always reflect future trends, especially during economic or market shifts.
This model only considers closing price data. Other factors can influence stock prices.

## Additional Considerations:
You can experiment with different hyperparameters (learning rate, number of layers, etc.) to potentially improve the model's performance.
Including additional features like trading volume or market sentiment might enhance the model's accuracy.
It's essential to evaluate the model's performance on unseen data (test set) to assess its generalizability.
