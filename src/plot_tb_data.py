import pandas as pd
import os
import plotly.express as px

data_source = "/Users/alexrutherford/Documents/4yp_plot_data"

file  = "supermarket-large-tc-2.csv"
file = "warehouse-large-5A-TC-TT.csv"

data = pd.read_csv(os.path.join(data_source, file))

print(data.head())


fig = px.line(data.iloc[0:500], x = 'Step', y = 'Value',
              width=800, height=800)
fig.add_hline(y=15.78355184, line_width=3, line_dash="dash", line_color="green")  # Charlie baseline
fig.add_hline(y=300, line_width=3, line_dash="dash", line_color="gray")  # Camas cutoff
fig.update_layout(template="simple_white")
fig.update_xaxes(showgrid=True, title_text="Training Timestep")
fig.update_yaxes(showgrid=True, title_text="Episode makespan (s)")
#fig.write_image("images/fig1.png") can also be svg or pdf

fig.show()