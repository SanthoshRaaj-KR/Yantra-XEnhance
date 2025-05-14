import React, { useState, useRef } from 'react';

const ImageDenoiser = () => {
  // State management
  const [image, setImage] = useState({
    file: null,
    previewUrl: '',
    processedUrl: '',
    noiseType: '',
  });
  const [status, setStatus] = useState({
    isLoading: false,
    error: '',
    message: '',
  });
  
  const fileInputRef = useRef(null);
  const API_URL = 'http://localhost:8000/api/denoise';

  // Handle file selection
  const handleFileSelection = (e) => {
    const selectedFile = e.target.files[0];
    
    if (!selectedFile) return;
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp'];
    if (!validTypes.includes(selectedFile.type)) {
      setStatus({
        ...status,
        error: 'Please select a valid image file (JPG, PNG, TIFF, BMP)',
        message: '',
      });
      return;
    }
    
    // Update state with new file
    setImage({
      file: selectedFile,
      previewUrl: URL.createObjectURL(selectedFile),
      processedUrl: '',
      noiseType: '',
    });
    
    setStatus({
      isLoading: false,
      error: '',
      message: '',
    });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent form submission from reloading the page
    
    if (!image.file) {
      setStatus({
        ...status,
        error: 'Please select an X-ray image first',
        message: '',
      });
      return;
    }

    // Show loading state
    setStatus({
      isLoading: true,
      error: '',
      message: 'Processing your image...',
    });

    try {
      // Create form data for API request
      const formData = new FormData();
      formData.append('file', image.file);

      // Send request to API
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      // Handle API error
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process image');
      }

      // Process successful response
      const data = await response.json();
      
      // Handle URL construction more robustly
      let fullUrl;
      if (data.processed_url.startsWith('http')) {
        // If it's already a full URL, use it as is
        fullUrl = data.processed_url;
      } else {
        // If it's a relative path, construct the full URL
        const pathWithSlash = data.processed_url.startsWith('/') 
          ? data.processed_url 
          : `/${data.processed_url}`;
        fullUrl = `http://localhost:8000${pathWithSlash}`;
      }

      console.log('Processed image URL:', fullUrl); // For debugging

      // Update state with processed image
      setImage({
        ...image,
        processedUrl: fullUrl,
        noiseType: data.noise_type,
      });
      
      setStatus({
        isLoading: false,
        error: '',
        message: 'Image successfully denoised!',
      });
    } catch (err) {
      // Handle errors
      setStatus({
        isLoading: false,
        error: err.message || 'Failed to process image',
        message: '',
      });
    }
  };

  // Handle download - using a custom function instead of directly using the anchor tag
  const handleDownload = (e) => {
    e.preventDefault(); // Prevent default navigation behavior
    
    if (!image.processedUrl) return;
    
    try {
      // Create a temporary anchor element
      const link = document.createElement('a');
      link.href = image.processedUrl;
      link.download = `denoised-xray-${new Date().getTime()}.png`;
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      console.log('Download initiated for:', image.processedUrl);
    } catch (err) {
      console.error('Download failed:', err);
      setStatus({
        ...status,
        error: 'Failed to download image. Please try again.',
      });
    }
  };

  // Reset the application state
  const handleReset = (e) => {
    e.preventDefault(); // Prevent any form submission
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    
    // Reset state
    setImage({
      file: null,
      previewUrl: '',
      processedUrl: '',
      noiseType: '',
    });
    
    setStatus({
      isLoading: false,
      error: '',
      message: '',
    });
  };

  return (
    <div className="container mx-auto p-6 max-w-5xl">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-gray-800">X-Ray Image Denoiser</h1>
        <p className="text-gray-600 mt-2">Upload and remove noise from X-ray images</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left panel - Upload */}
        <section className="bg-white rounded-xl shadow-md overflow-hidden">
          <div className="p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Upload X-Ray Image</h2>
            
            <form onSubmit={handleSubmit}>
              {/* File input */}
              <div className="mb-6">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileSelection}
                  accept="image/jpeg,image/png,image/tiff,image/bmp"
                  className="hidden"
                  id="file-upload"
                />
                
                {!image.previewUrl ? (
                  <label
                    htmlFor="file-upload"
                    className="border-2 border-dashed border-gray-300 rounded-lg p-8 flex flex-col items-center justify-center cursor-pointer hover:border-blue-500 transition-colors"
                  >
                    <svg 
                      className="w-12 h-12 text-gray-400 mb-3" 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24" 
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth="2" 
                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                      ></path>
                    </svg>
                    <span className="text-gray-600">Click to select an X-Ray image</span>
                    <span className="text-sm text-gray-400 mt-2">Supports JPG, PNG, TIFF, BMP</span>
                  </label>
                ) : (
                  <div className="relative">
                    <img
                      src={image.previewUrl}
                      alt="Preview"
                      className="w-full h-auto rounded-lg border border-gray-200"
                    />
                    <button
                      type="button"
                      onClick={handleReset}
                      className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full hover:bg-red-600"
                      aria-label="Remove image"
                    >
                      ✕
                    </button>
                  </div>
                )}
              </div>

              {/* Denoise button */}
              {image.previewUrl && (
                <button
                  type="submit"
                  disabled={status.isLoading}
                  className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
                    status.isLoading
                      ? 'bg-blue-400 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }`}
                >
                  {status.isLoading ? 'Processing...' : 'Denoise Image'}
                </button>
              )}
              
              {/* Status messages */}
              {status.message && (
                <div className="mt-4 p-3 bg-green-100 text-green-700 rounded-lg">
                  {status.message}
                </div>
              )}
              
              {status.error && (
                <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg">
                  {status.error}
                </div>
              )}
            </form>
          </div>
        </section>

        {/* Right panel - Results */}
        <section className="bg-white rounded-xl shadow-md overflow-hidden">
          <div className="p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Results</h2>
            
            {image.processedUrl ? (
              <div>
                <div className="mb-4">
                  <p className="text-gray-700 mb-2">
                    <span className="font-medium">Detected Noise Type:</span>{' '}
                    {image.noiseType && image.noiseType.charAt(0).toUpperCase() + image.noiseType.slice(1)} Noise
                  </p>
                  
                  <div className="relative pt-4">
                    <img
                      src={image.processedUrl}
                      alt="Denoised result"
                      className="w-full h-auto rounded-lg border border-gray-200"
                      onError={() => {
                        setStatus({
                          ...status,
                          error: 'Failed to load processed image'
                        });
                      }}
                    />
                    <div className="absolute top-0 left-0 bg-green-100 text-green-800 rounded-tr rounded-bl px-2 py-1 text-xs font-medium">
                      Denoised
                    </div>
                  </div>
                </div>
                
                <div className="flex gap-3 mt-6">
                  <button
                    onClick={handleDownload}
                    className="flex-1 py-2 px-4 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    Download
                  </button>
                  <button
                    onClick={handleReset}
                    className="flex-1 py-2 px-4 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
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
        </section>
      </div>
    </div>
  );
};

export default ImageDenoiser;