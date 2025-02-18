import os
from graphviz import Digraph

# Add Graphviz path to system PATH (if not already added globally)
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

# Create a new directed graph
er = Digraph('ER_Model', format='png', engine='dot')
er.attr(rankdir='LR', size='10')

# Define entities with their attributes
entities = {
    "Flight": ["id (PK)", "codigo_vuelo (UQ)", "aerolinea", "origen", "destino", 
               "fecha_hora_salida", "fecha_hora_llegada", "capacidad_total", 
               "asientos_disponibles", "estado", "precio_base", "tipo_avion"],
    
    "Seat": ["id (PK)", "numero_asiento", "clase", "estado", "vuelo_id (FK)"],
    
    "User": ["id (PK)", "nombre", "apellido", "email (UQ)", "telefono", 
             "documento_identidad (UQ)", "tipo_documento", "nacionalidad", 
             "fecha_nacimiento", "fecha_registro", "rol"],
    
    "Reservation": ["id (PK)", "usuario_id (FK)", "vuelo_id (FK)", "asiento_id (FK)", 
                    "estado", "fecha_reserva", "precio_final", "codigo_reserva (UQ)"],
    
    "Payment": ["id (PK)", "reserva_id (FK)", "monto", "fecha_pago", "metodo_pago", 
                "estado", "transaccion_id (UQ)"],
    
    "Notification": ["id (PK)", "usuario_id (FK)", "reserva_id (FK)", "tipo_notificacion", 
                     "estado_envio", "fecha_envio"]
}

# Add entity nodes to the graph
for entity, attributes in entities.items():
    label = f"{entity}| " + r"\n".join(attributes)
    er.node(entity, shape='record', label="{" + label + "}")

# Define relationships between entities
relations = [
    ("Flight", "Seat", "1", "N"),  # A flight has many seats
    ("Flight", "Reservation", "1", "N"),  # A flight has many reservations
    ("Seat", "Reservation", "1", "1"),  # A seat can only have one reservation
    ("User", "Reservation", "1", "N"),  # A user can make many reservations
    ("Reservation", "Payment", "1", "1"),  # A reservation has one payment
    ("User", "Notification", "1", "N"),  # A user can receive many notifications
    ("Reservation", "Notification", "1", "N")  # A reservation generates many notifications
]

# Add relationships to the graph
for src, dest, min_card, max_card in relations:
    er.edge(src, dest, label=f"{min_card}:{max_card}")

# Save the graph as a PNG image
er.render("ER_Model", format="png", cleanup=True)

# Print the path where the image is saved
print("ER Diagram saved as ER_Model.png")
