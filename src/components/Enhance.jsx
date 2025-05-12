import React, { useState, useRef } from 'react';

const ImageDenoiser = () => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [processedUrl, setProcessedUrl] = useState('');
  const [noiseType, setNoiseType] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const fileInputRef = useRef(null);

  const API_URL = 'http://localhost:8000/api/denoise';

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      // Validate file type
      const validTypes = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp'];
      if (!validTypes.includes(selectedFile.type)) {
        setError('Please select a valid image file (JPG, PNG, TIFF, BMP)');
        return;
      }
      
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setProcessedUrl('');
      setNoiseType('');
      setError('');
      setMessage('');
    }
  };

  const handleSubmit = async (e) => {
    // Prevent form submission default behavior
    e.preventDefault();
    
    if (!file) {
      setError('Please select an X-ray image first');
      return;
    }

    setIsLoading(true);
    setError('');
    setMessage('Processing your image...');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process image');
      }

      const data = await response.json();
      setProcessedUrl(`http://localhost:8000${data.processed_url}`);
      setNoiseType(data.noise_type);
      setMessage('Image successfully denoised!');
    } catch (err) {
      setError(err.message || 'Failed to process image');
      setMessage('');
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setPreviewUrl('');
    setProcessedUrl('');
    setNoiseType('');
    setError('');
    setMessage('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-3xl font-bold text-center mb-8">X-Ray Image Denoiser</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Upload X-Ray Image</h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              className="hidden"
            />
            
            {!previewUrl ? (
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
                onClick={() => fileInputRef.current.click()}
              >
                <svg 
                  className="mx-auto h-12 w-12 text-gray-400" 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="mt-1 text-gray-600">Click to select an X-Ray image</p>
                <p className="text-sm text-gray-400 mt-2">Supports JPG, PNG, TIFF, BMP</p>
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
                    type="button"
                    onClick={resetForm}
                    className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full hover:bg-red-600 focus:outline-none"
                  >
                    ✕
                  </button>
                </div>
                
                <button
                  type="submit"
                  disabled={isLoading}
                  className={`w-full py-2 px-4 rounded-lg text-white font-medium transition-colors ${
                    isLoading ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
                  }`}
                >
                  {isLoading ? 'Processing...' : 'Denoise Image'}
                </button>
              </div>
            )}
            
            {message && (
              <div className="mt-4 p-3 bg-green-100 text-green-700 rounded-lg">
                {message}
              </div>
            )}
            
            {error && (
              <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg">
                {error}
              </div>
            )}
          </form>
        </div>

        {/* Results Section */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Results</h2>
          
          {processedUrl ? (
            <div className="space-y-4">
              <div>
                <p className="mb-2 text-gray-700">
                  <span className="font-medium">Detected Noise Type:</span> {noiseType && noiseType.charAt(0).toUpperCase() + noiseType.slice(1)} Noise
                </p>
                <div className="relative pt-4">
                  <img
                    src={processedUrl}
                    alt="Denoised result"
                    className="w-full h-auto rounded-lg border border-gray-200"
                    onError={() => setError('Failed to load processed image')}
                  />
                  <div className="absolute top-0 left-0 bg-green-100 text-green-800 rounded-tr rounded-bl px-2 py-1 text-xs font-medium">
                    Denoised
                  </div>
                </div>
              </div>
              
              <div className="flex space-x-3">
                <a
                  href={processedUrl}
                  download="denoised-xray.png"
                  className="flex-1 py-2 px-4 bg-green-600 text-white rounded-lg text-center hover:bg-green-700 transition-colors focus:outline-none"
                >
                  Download
                </a>
                <button
                  type="button"
                  onClick={resetForm}
                  className="flex-1 py-2 px-4 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors focus:outline-none"
                >
                  New Image
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-gray-100 rounded-lg p-8 text-center h-64 flex items-center justify-center">
              <p className="text-gray-500">Denoised image will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageDenoiser;