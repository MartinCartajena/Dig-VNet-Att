



import pandas as pd
import matplotlib.pyplot as plt

# Configurar matplotlib para usar backend 'Agg' (sin interfaz gráfica)
plt.switch_backend('Agg')

# Nombres de los archivos CSV
train_file = '/home/VICOMTECH/mcartajena/LUCIA/Dig-CS-VNet/vnet.pytorch/results/logs/train_20241118_092133.csv'
validation_file = '/home/VICOMTECH/mcartajena/LUCIA/Dig-CS-VNet/vnet.pytorch/results/logs/validation_20241118_092133.csv'

# Cargar los datos de los archivos CSV y verificar si se leen correctamente
train_data = pd.read_csv(train_file, header=None, names=['epoch', 'softdice'])
validation_data = pd.read_csv(validation_file, header=None, names=['epoch', 'softdice'])

min_validation_data = min(round(validation_data['softdice'], 3))

# Imprimir los primeros datos para verificar que se han cargado correctamente
print("Train Data Head:\n", train_data.head())
print("Validation Data Head:\n", validation_data.head())

# Crear el gráfico de líneas con ambas líneas
plt.figure(figsize=(10, 6))
plt.plot(train_data['epoch'].values, train_data['softdice'].values, label='Train', color='blue')
plt.plot(validation_data['epoch'].values, validation_data['softdice'].values, label='Validation', color='orange')

# Configurar los detalles del gráfico
plt.xlabel('Epoch')
plt.ylabel('Softdice')
plt.title(f'Loss Graph / Data Aug & Dig_Sep / Best Val: {min_validation_data}')
plt.legend()
plt.grid(True)

# Guardar el gráfico como imagen
plt.savefig('/home/VICOMTECH/mcartajena/LUCIA/Dig-CS-VNet/vnet.pytorch/results/plots/softdice_per_epoch_20241118_092133.png')
print("El gráfico ha sido guardado como 'softdice_per_epoch.png'")
