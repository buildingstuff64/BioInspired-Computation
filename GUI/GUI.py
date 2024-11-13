import json

from nicegui import ui, binding, Client, app, run
from nicegui.binding import BindableProperty
from nicegui.events import ValueChangeEventArguments
from multiprocessing import Manager, Queue
import numpy as np
import pandas as pd

from Implementation.PSO import PSO

progress = BindableProperty()

def show(event: ValueChangeEventArguments):
    name = type(event.sender).__name__
    ui.notify(f'{name}: {event.value}')

def trainNNN(q :Queue):
    file = pd.read_csv('../Data/concrete_data.csv')
    input_data = np.array(file.iloc[:, : 8].to_numpy())
    output_data = np.array(file.iloc[:, 8:].to_numpy())

    input_data_train = input_data[int(len(input_data) * 0.7):]
    output_data_train = output_data[int(len(output_data) * 0.7):]

    input_data_test = input_data[-int(len(input_data) * 0.3):]
    output_data_test = output_data[-int(len(output_data) * 0.3):]

    pso_data = json.load(open('../Data/hyperparameters.json', 'r'))
    pso = PSO(pso_data, (input_data_train, output_data_train), (input_data_test, output_data_test))
    best_position, l = pso.optimise(q=q)
    return l

async def computeTrain(queue):
    ui.notify(f'Started Training...')
    losses = await run.cpu_bound(trainNNN, queue)
    queue.put_nowait(1)
    draw_fig.refresh(losses)
    ui.notify('Finished')

def saveData(data):
    print("save data")
    data['weight_range'] = list(data['weight_range'].values())
    data['bias_range'] = list(data['bias_range'].values())
    data['swarm_size'] = int(data['swarm_size'])
    data['iterations'] = int(data['iterations'])
    data['informants'] = int(data['informants'])
    print(data)
    with open('../Data/hyperparameters.json', 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))
    ui.notify(f'JSON Data Successfully Saved')

def save_train(data, queue):
    saveData(data)
    computeTrain(queue)


@ui.refreshable
def draw_fig(losses):
    with ui.matplotlib(figsize = (14, 8)).figure as fig:
        ax = fig.gca()
        ax.set_title(f"Loss Over Time        Final Loss : {losses[-1] if len(losses)>0 else 0}")
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.plot(losses)

@ui.page('/')
async def index(client: Client):
    pso_data = json.load(open('../Data/hyperparameters.json', 'r'))

    with ui.row():
        with ui.card():

            data = {}
            losses = []

            ui.label(text="Hyperparameters for PSO Algorithm")
            ui.link('github repo', 'https://github.com/buildingstuff64/BioInspired-Computation', new_tab = True)

            swarm_size = ui.number(label="Swarm Size", value=pso_data['swarm_size'], format='%1d').bind_value_to(data, 'swarm_size')
            iterations = ui.number(label="Iterations", value=pso_data['iterations'], format='%1d').bind_value_to(data, 'iterations')
            ui.separator()

            layer = ui.input(label = 'Layer Dimensions', placeholder = 'format [input, hidden..., output]', value = pso_data['layer_sizes']).bind_value_to(data, 'layer_sizes')
            activation = ui.select(label= 'Activation Function', options = ['sigmoid', 'ReLU', 'tanh'],
                                   with_input = True, value = pso_data['activation']).bind_value_to(data, 'activation')
            ui.separator()

            inform = ui.number(label = "Informants", value = pso_data['informants'], format = '%1d').bind_value_to(data, 'informants')
            a = ui.number(label = "Alpha", value = pso_data['alpha'], format = '%.2f').bind_value_to(data, 'alpha')
            b = ui.number(label = "Beta", value = pso_data['beta'], format = '%.2f').bind_value_to(data, 'beta')
            g = ui.number(label = "Gamma", value = pso_data['gamma'], format = '%.2f').bind_value_to(data, 'gamma')
            d = ui.number(label = "Delta", value = pso_data['delta'], format = '%.2f').bind_value_to(data, 'delta')
            ui.separator()

            adv = ui.checkbox('Advanced Settings', value=False)
            with ui.card().bind_visibility_from(adv, 'value'):
                weight_range = ui.range(min = -2, max = 2, value = {'min': pso_data['weight_range'][0],
                                                                    'max': pso_data['weight_range'][1]},
                                        step = 0.01).bind_value_to(data, 'weight_range')
                ui.label().bind_text_from(weight_range, 'value', backward = lambda v: f'Initialization Weight Ranges : {v["min"]:<5},{v["max"]:<5}')

                bias_range = ui.range(min = -2, max = 2,
                                      value = {'min': pso_data['bias_range'][0], 'max': pso_data['bias_range'][1]},
                                      step = 0.01).bind_value_to(data, 'bias_range')
                ui.label().bind_text_from(bias_range, 'value', backward = lambda v: f'Initialization Bias Ranges : {v["min"]:<5} , {v["max"]:<5}')



            with ui.button_group():
                ui.button('Train', on_click = lambda: computeTrain(queue))
                ui.button('Save', on_click = lambda: saveData(data))
                ui.button('Save & Train', on_click = lambda: save_train(data, queue))

        with ui.card():
            draw_fig(losses)

            queue = Manager().Queue()
            ui.separator()

            ui.timer(0.01,callback = lambda: progressbar.set_value(queue.get() if not queue.empty() else progressbar.value))
            progressbar = ui.linear_progress(value=0, show_value = False, size = "25px").props('instant-feedback')


        await client.connected()
        print("connected to local server")
        await  client.disconnected()
        print("disconnected : shutting app down...")
        app.shutdown()

if __name__ == '__main__':
    ui.run(reload = False, title = "Particle Swarm Optimization")