class Lane:
    def __init__(self):
        self.solid = 'Solid'		# Solid/dotted
        self.color = 'Yellow'       # Color
        self.single = 'Single'		# Single/Double

    def __str__(self):
        return self.solid + ' ' + self.single + ' ' + self.color
