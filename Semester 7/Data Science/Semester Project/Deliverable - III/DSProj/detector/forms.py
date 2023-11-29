# forms.py
from django import forms


class MyForm(forms.Form):
    step = forms.IntegerField(
        widget=forms.TextInput(
            attrs={
                'type': 'number',
                'id' : 'step',
                'class' : "bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 "
                          "focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 "
                          "dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 "
                          "dark:focus:border-blue-500",
                'placeholder': ' ',
                'required' : 'required',
                'min': 0,
                'max': 1000,
                'step': 1
            }
        )
    )
    amount = forms.DecimalField(
        widget=forms.TextInput(
            attrs={
                'type': 'number',
                'id': 'amount',
                'class': "bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 "
                         "focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 "
                         "dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 "
                         "dark:focus:border-blue-500",
                'placeholder': ' ',
                'required': 'required',
                'min': 0,
                'max': 100000000,
                'step': 0.01
            }
        )
    )
    oldbalanceOrg = forms.DecimalField(
        widget=forms.TextInput(
            attrs={
                'type': 'number',
                'id': 'amount',
                'class': "bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 "
                         "focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 "
                         "dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 "
                         "dark:focus:border-blue-500",
                'placeholder': ' ',
                'required': 'required',
                'min': 0,
                'max': 100000000,
                'step': 0.01
            }
        )
    )
    newbalanceOrig = forms.DecimalField(
        widget=forms.TextInput(
            attrs={
                'type': 'number',
                'id': 'amount',
                'class': "bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 "
                         "focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 "
                         "dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 "
                         "dark:focus:border-blue-500",
                'placeholder': ' ',
                'required': 'required',
                'min': 0,
                'max': 100000000,
                'step': 0.01
            }
        )
    )
    oldbalanceDest = forms.DecimalField(
        widget=forms.TextInput(
            attrs={
                'type': 'number',
                'id': 'amount',
                'class': "bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 "
                         "focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 "
                         "dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 "
                         "dark:focus:border-blue-500",
                'placeholder': ' ',
                'required': 'required',
                'min': 0,
                'max': 100000000,
                'step': 0.01
            }
        )
    )
    newbalanceDest = forms.DecimalField(
        widget=forms.TextInput(
            attrs={
                'type': 'number',
                'id': 'amount',
                'class': "bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 "
                         "focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 "
                         "dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 "
                         "dark:focus:border-blue-500",
                'placeholder': ' ',
                'required': 'required',
                'min': 0,
                'max': 10000000,
                'step': 0.01
            }
        )
    )
    isFlaggedFraud = forms.BooleanField(required=False)  # Checkbox field
    transaction_type = forms.ChoiceField(
        choices=[
            ('CASH_OUT', 'Cash Out'),
            ('DEBIT', 'Debit'),
            ('PAYMENT', 'Payment'),
            ('TRANSFER', 'Transfer'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )