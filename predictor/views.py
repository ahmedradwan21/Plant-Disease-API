from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import logging
from .services import predict_image
from .serializers import PredictionSerializer

logging.basicConfig(level=logging.INFO)

class PredictView(APIView):

    def post(self, request, *args, **kwargs):
        file = request.FILES.get('file')
        
        if not file:
            return Response({'error': 'No file part in the request'}, status=status.HTTP_400_BAD_REQUEST)
        
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.name)
        
        with open(temp_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        try:
            result = predict_image(temp_path)
            os.remove(temp_path)
            serializer = PredictionSerializer(data={'prediction': result})
            if serializer.is_valid():
                return Response(serializer.data)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        except Exception as e:
            logging.error(f'Error during prediction: {e}')
            return Response({'error': 'An error occurred during prediction'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
