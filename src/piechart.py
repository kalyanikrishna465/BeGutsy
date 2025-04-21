import plotly.express as px

# Sample data
labels = ['Healthy', 'Unhealthy']
values = [3046, 2688]

# Create pie chart
fig = px.pie(
    names=labels,
    values=values,
    title='Overall Sample Distribution: Healthy vs. Unhealthy',
    color=labels,
    color_discrete_map={'Healthy': '#28a745', 'Unhealthy': '#dc3545'}
)

# Customize layout
fig.update_traces(textinfo='percent+label', pull=[0.05, 0], hole=0.3)
fig.update_layout(title_x=0.5)

# Save as PNG
fig.write_image("pie_chart.png", format="png", scale=2)
fig.show()
