from dill import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.shortcuts import render
from .forms import MyForm
from .predictor import predictor

# Create your views here.
def index(request):
    form = MyForm()
    return render(request, 'detector/index.html', {'form': form})


def home(request):
    return render(request, "detector/home.html")


def transaction_form(request):
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            # Process the form data
            step = form.cleaned_data['step']
            amount = float(form.cleaned_data['amount'])
            oldbalanceOrg = float(form.cleaned_data['oldbalanceOrg'])
            newbalanceOrig = float(form.cleaned_data['newbalanceOrig'])
            oldbalanceDest = float(form.cleaned_data['oldbalanceDest'])
            newbalanceDest = float(form.cleaned_data['newbalanceDest'])
            isFlaggedFraud = float(form.cleaned_data['isFlaggedFraud'])
            transaction_type = form.cleaned_data['transaction_type']

            # Get the Transaction Type Part such that it can be given to the model just like when it was given in
            # model training
            possible_types = ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

            # Initialize the list with zeros
            type_parameters = [0] * len(possible_types)

            # Find the index of the selected type and set it to 1
            if transaction_type in possible_types:
                i = possible_types.index(transaction_type)
                type_parameters[i] = 1

            features = [step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFlaggedFraud]
            features.extend(type_parameters)

            result = predictor(features)

            return render(request, 'detector/result.html', {
                'step': step,
                'amount': amount,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrig,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest,
                'isFlaggedFraud': isFlaggedFraud,
                'transaction_type': transaction_type,
                'result': result
            })
    else:
        form = MyForm()

    return render(request, 'my_form.html', {'form': form})
