from django.shortcuts import render
from src.predict_pipeline import run_prediction


def predict_view(request):
    if request.method == "POST":
        stock = request.POST.get("stock")
        model = request.POST.get("model")

        result = run_prediction(stock, model)

        return render(request, "predict.html", {
            "result": result
        })

    return render(request, "predict.html")