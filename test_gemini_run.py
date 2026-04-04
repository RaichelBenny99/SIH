from services.gemini import ask_agronomist

result = ask_agronomist(
    'Tomato - Septoria leaf spot',
    {'severity': 'Severe', 'infected_pct': 80},
    {'temperature': 28, 'humidity': 70, 'risk': 'High'},
    'What is best treatment?'
)
print(result)
