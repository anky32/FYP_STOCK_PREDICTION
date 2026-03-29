from django.db import models


class Feedback(models.Model):
    stock = models.CharField(max_length=10)
    model = models.CharField(max_length=10)
    rating = models.IntegerField()
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.stock} - {self.rating}"