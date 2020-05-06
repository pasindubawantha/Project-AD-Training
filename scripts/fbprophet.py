import pandas as pd
import numpy as np
import fbprophet 
# import Prophet

input_dataframe = pd.read_csv('../data/lseg/lse_2009-07-24_2020-03-20.csv')


data = {'y':np.array(input_dataframe['Open'])}
df = pd.DataFrame(data, index=np.array(input_dataframe['timestamp']))
df.index.name = "ds"

df.head()

m = fbprophet.Prophet()
m.fit(df)

# future = m.make_future_dataframe(periods=365)
# future.tail()

# forecast = m.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# fig1 = m.plot(forecast)

# fig2 = m.plot_components(forecast)


# from fbprophet.plot import plot_plotly
# import plotly.offline as py
# py.init_notebook_mode()

# fig = plot_plotly(m, forecast)  # This returns a plotly Figure
# py.iplot(fig)