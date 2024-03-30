import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functools import reduce

color = '30, 144, 255'
color = '33, 150, 243'
color_theme_plot = '#2196f3'

class NeuralNetworkPlot:
	def __init__(self, 
		architecture: tuple[int], 
		layer_distance:  float= 10.,
		neuron_distance: float=10.,
		neuron_size: float=15.):
		
		self.architecture = architecture
		self.layer_distance = layer_distance
		self.neuron_distance = neuron_distance
		self.neuron_size = neuron_size

		self.n_layers = len(self.architecture)
		self.max_neurons_in_layer = max(self.architecture)
		self.total_paths = np.prod(self.architecture)

	def get_neuron_heights(self, n_neurons: int):
		half_n_neurons = n_neurons // 2
		starter_y = -self.neuron_distance * (half_n_neurons - (0. if n_neurons%2==1 else 0.5))
		return [starter_y + i * self.neuron_distance for i in range(n_neurons)]

	def get_layer_coords(self, index_layer: int):
		n_neurons = self.architecture[index_layer]
		layer_x = index_layer * self.layer_distance
		return [[layer_x, layer_y]
			for layer_y in self.get_neuron_heights(n_neurons)
		]

	def get_coords(self):
		return np.array(reduce(
			lambda x,y : x+y, 
			[self.get_layer_coords(index_layer) for index_layer in range(self.n_layers)]
		))

	def fig(self):

		fig = go.Figure()

		fig.add_trace(go.Scatter(
			mode='markers',
			x=self.get_coords()[:, 0],
			y=self.get_coords()[:, 1],
			marker=dict(
				color=f'rgba({color},1)',
				size=self.neuron_size,
				line=dict(color=f'rgb({color})', width=2)
			),
		))

		for index_layer in range(self.n_layers - 1):
			for point_a in self.get_layer_coords(index_layer):
				for point_b in self.get_layer_coords(index_layer+1):
					fig.add_trace(go.Scatter(
						x=[point_a[0], point_b[0]],
						y=[point_a[1], point_b[1]],
						line=go.scatter.Line(color=f'rgba({color},0.5)', width=1.),
						marker=go.scatter.Marker(color=f'rgba({color},0)')
					))
		
		fig.update_layout(
			showlegend=False,
			margin=dict(l=0, r=0, t=0, b=0),
			width=700,
			height=(550 // 8) * self.max_neurons_in_layer,
		)
		fig.update_xaxes(visible=False)
		fig.update_yaxes(visible=False)

		return fig

def update_nn_fig_hoverlabel(
	fig: go.Figure, 
	encoder_arch: tuple[int], 
	decoder_arch: tuple[int], 
	n_bottleneck_neurons: int ) -> go.Figure:
	
	list_labels = (
		  ['Input Y', 'Input X'] 
		+ ['Neuron'] * sum(encoder_arch)
		+ (['Bottleneck'] if n_bottleneck_neurons == 1 else ['Bottleneck 2', 'Bottleneck 1'])
		+ ['Neuron'] * sum(decoder_arch)
		+ ['Output Y', 'Output X']
	)

	fig.update_traces(
		patch={'hoverinfo' : 'skip'},
		selector=lambda trace : trace != 'trace 0'
	)

	fig.data[0].update({
		'hoverinfo' : 'text',
		'hovertext' : list_labels
	})

	fig.update_layout(
		hoverlabel=dict(
			bgcolor=color_theme_plot,
			bordercolor=color_theme_plot,
			font_size=16,
		),
	)

	return fig
