import React, { useState, useRef } from 'react';
import axios from 'axios';

const ImageDenoiser = () => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [processedUrl, setProcessedUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef();

  const API_URL = 'http://localhost:8000/api/denoise';

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setProcessedUrl('');
      setError('');
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      setError('Please select an image first');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setProcessedUrl(`http://localhost:8000${response.data.processed_url}`);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to process image');
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setPreviewUrl('');
    setProcessedUrl('');
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-3xl font-bold text-center mb-8">Image Denoiser</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Upload Image</h2>
          
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            className="hidden"
          />
          
          {!previewUrl ? (
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer"
                 onClick={() => fileInputRef.current.click()}>
              <p className="text-gray-500">Click to select an image</p>
              <p className="text-sm text-gray-400 mt-2">Supports JPG, PNG</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative">
                <img 
                  src={previewUrl} 
                  alt="Preview" 
                  className="w-full h-auto rounded-lg border border-gray-200"
                />
                <button 
                  onClick={resetForm}
                  className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full hover:bg-red-600"
                >
                  ✕
                </button>
              </div>
              
              <button
                onClick={handleSubmit}
                disabled={isLoading}
                className={`w-full py-2 px-4 rounded-lg text-white font-medium ${
                  isLoading ? 'bg-blue-400' : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {isLoading ? 'Processing...' : 'Denoise Image'}
              </button>
            </div>
          )}
          
          {error && (
            <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg">
              {error}
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Results</h2>
          
          {processedUrl ? (
            <div className="space-y-4">
              <img
                src={processedUrl}
                alt="Denoised result"
                className="w-full h-auto rounded-lg border border-gray-200"
                onError={() => setError('Failed to load processed image')}
              />
              
              <div className="flex space-x-3">
                <a
                  href={processedUrl}
                  download="denoised-image.png"
                  className="flex-1 py-2 px-4 bg-green-600 text-white rounded-lg text-center hover:bg-green-700"
                >
                  Download
                </a>
                <button
                  onClick={resetForm}
                  className="flex-1 py-2 px-4 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
                >
                  New Image
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-gray-100 rounded-lg p-8 text-center">
              <p className="text-gray-500">Processed image will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageDenoiser;