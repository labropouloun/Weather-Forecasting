\# Weather-Forecasting



This project performs weather forecasting for the maximum temperature in Athens, Greece, using various time series forecasting techniques. The forecasts are based on two years of daily data (2021-2022) and include comparisons with other Greek locations to enhance predictions.



\## Project Description



The project workflow includes:



1\. \*\*Data Collection\*\*  

&nbsp;  - Daily maximum temperature data for Athens from 01/01/2021 to 31/12/2022.

&nbsp;  - Additional datasets for 10 other Greek locations (Alexandroupoli, Corfu, Crete, Kalamata, Karpenisi, Kavala, Samos, Thessaloniki, Xanthi, Zakynthos) for comparison and enhanced forecasting.



2\. \*\*Data Cleaning\*\*  

&nbsp;  - Remove inconsistencies, outliers, and missing values in the datasets.



3\. \*\*Forecasting Techniques\*\*

&nbsp;  - \*\*5-Day Moving Average\*\*

&nbsp;  - \*\*5-Day Weighted Moving Average\*\*

&nbsp;  - \*\*7-Day Moving Average\*\*

&nbsp;  - \*\*7-Day Weighted Moving Average\*\*

&nbsp;  - \*\*Exponential Smoothing\*\* (optimal alpha selected based on MSE and MAD)

&nbsp;  - \*\*Holt Linear Trend Method\*\*

&nbsp;  - \*\*Holt-Winters Seasonal Method\*\* (trend and seasonality adjusted)



4\. \*\*Evaluation Metrics\*\*

&nbsp;  - Mean Squared Error (MSE)

&nbsp;  - Mean Absolute Deviation (MAD)



5\. \*\*Comparison\*\*

&nbsp;  - Forecasts for both 2021 and 2022 are compared.

&nbsp;  - Forecast for Athens first day of each month in 2021 is enhanced using data from 10 similar locations.



6\. \*\*Output\*\*

&nbsp;  - Forecast results saved as CSV files for each method.

&nbsp;  - Plots saved as PNG images for visualization of actual vs forecasted temperatures.





\## Project Structure



Weather-Forecasting/

-- code.py # main program

-- requirements.txt

-- README.md

-- data/

&nbsp;  -- Athens 2021-01-01 to 2023-01-01.csv

&nbsp;  -- Alexandroupoli 2021-01-01 to 2022-12-31.csv

&nbsp;  -- Corfu Greece 2021-01-01 to 2022-12-31.csv

&nbsp;  -- Crete Greece 2021-01-01 to 2022-12-31.csv

&nbsp;  -- Kalamata Greece 2021-01-01 to 2022-12-31.csv

&nbsp;  -- Karpenisi Greece 2021-01-01 to 2022-12-31.csv

&nbsp;  -- Kavala, Greece 2021-01-01 to 2022-12-31.csv

&nbsp;  -- Samos Greece 2021-01-01 to 2022-12-31.csv

&nbsp;  -- Thessaloniki 2021-01-01 to 2022-12-31.csv

&nbsp;  -- Xanthi Greece 2021-01-01 to 2022-12-31.csv

&nbsp;  -- Zakynthos Greece 2021-01-01 to 2022-12-31.csv

-- results\_csv/         # All forecast results

&nbsp;  -- 5-DAY\_Moving\_Average\_forecasts\_2021.csv

&nbsp;  -- 5-DAY\_Weighted\_Moving\_Average\_forecasts\_2021.csv

&nbsp;  -- 7-DAY\_Moving\_Average\_forecasts\_2021.csv

&nbsp;  -- 7-DAY\_Weighted\_Moving\_Average\_forecasts\_2021.csv

&nbsp;  -- 5-DAY\_Moving\_Average\_forecasts\_2022.csv

&nbsp;  -- 5-DAY\_Weighted\_Moving\_Average\_forecasts\_2022.csv

&nbsp;  -- 7-DAY\_Moving\_Average\_forecasts\_2022.csv

&nbsp;  -- 7-DAY\_Weighted\_Moving\_Average\_forecasts\_2022.csv

&nbsp;  -- Forecast\_2021\_MAD\_Exponential\_Smoothing.csv

&nbsp;  -- Forecast\_2022\_MAD\_Exponential\_Smoothing.csv

&nbsp;  -- Forecast\_2021\_MSE\_Exponential\_Smoothing.csv

&nbsp;  -- Forecast\_2022\_MSE\_Exponential\_Smoothing.csv

&nbsp;  -- Holt\_Linear\_Forecasts\_2021\_MAD.csv

&nbsp;  -- Holt\_Linear\_Forecasts\_2021\_MSE.csv

&nbsp;  -- Holt\_Linear\_Forecasts\_2022\_MAD.csv

&nbsp;  -- Holt\_Linear\_Forecasts\_2022\_MSE.csv

&nbsp;  -- Holt\_Winter\_Forecasts\_2021\_2022\_Whole\_Year.csv

-- results\_plots/        # All generated plots

&nbsp;  -- actual\_tempmax2021.png

&nbsp;  -- actual\_tempmax2022.png

&nbsp;  -- 5day\_weighted\_moving\_average\_2021.png

&nbsp;  -- 5day\_weighted\_moving\_average\_2022.png

&nbsp;  -- 7day\_weighted\_moving\_average\_2021.png

&nbsp;  -- 7day\_weighted\_moving\_average\_2022.png

&nbsp;  -- Combined\_Holt\_2021.png

&nbsp;  -- Combined\_Holt\_2022.png

&nbsp;  -- Exponential\_Smoothing\_2021\_MAD.png

&nbsp;  -- Exponential\_Smoothing\_2021\_MSE.png

&nbsp;  -- Exponential\_Smoothing\_2022\_MAD.png

&nbsp;  -- Exponential\_Smoothing\_2022\_MSE.png

&nbsp;  -- Holt's Winter Forecast\_vs\_Actual\_2021.png

&nbsp;  -- Holt's Winter Forecast\_vs\_Actual\_2022.png

&nbsp;  -- Holt\_Linear\_Forecasts\_2021.png

&nbsp;  -- Holt\_Linear\_Forecasts\_2022.png

&nbsp;  -- tempmax\_and\_5day\_moving\_average2021.png

&nbsp;  -- tempmax\_and\_7day\_moving\_average2021.png

&nbsp;  -- tempmax\_and\_5day\_moving\_average2022.png

&nbsp;  -- tempmax\_and\_7day\_moving\_average2022.png



\## Installation



Clone the repository:



git clone https://github.com/labropouloun/Weather-Forecasting.git

cd Weather-Forecasting

pip install -r requirements.txt



\# Run the main program:



python code.py



\## Libraries Used



pandas

numpy

matplotlib

statsmodels



\## Author

Nancy Labropoulou

GitHub: labropouloun





