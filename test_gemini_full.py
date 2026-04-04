from services.gemini import ask_agronomist

res = ask_agronomist(
    disease='Tomato - Septoria leaf spot',
    severity={'severity': 'Severe', 'infected_pct': 80},
    weather={'temperature': 23.1, 'humidity': 73, 'risk': 'Low'},
    user_question='What is the best treatment approach for this condition?'
)
print('Model:', res['model'])
print('Error:', res['error'])
print('\nFull Response:')
print(res['response'])
