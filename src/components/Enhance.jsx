import React, { useState, useRef } from 'react';


// The component is named Enhance to match the file name and usage in App.jsx
const Enhance = () => {
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
  // Ensure this URL matches your running Python backend
  const API_URL = 'https://santy171710-classifier.hf.space/api/denois';

  // Handle file selection
  const handleFileSelection = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    
    const validTypes = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp'];
    if (!validTypes.includes(selectedFile.type)) {
      setStatus({ ...status, error: 'Please select a valid image file (JPG, PNG, TIFF, BMP)', message: '' });
      return;
    }
    
    setImage({ file: selectedFile, previewUrl: URL.createObjectURL(selectedFile), processedUrl: '', noiseType: '' });
    setStatus({ isLoading: false, error: '', message: '' });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image.file) {
      setStatus({ ...status, error: 'Please select an X-ray image first', message: '' });
      return;
    }

    setStatus({ isLoading: true, error: '', message: 'Processing your image...' });
    const formData = new FormData();
    // The Python backend expects the key 'image'
    formData.append('image', image.file);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // Try to parse error from backend, otherwise show generic message
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.error || `Request failed with status ${response.status}`);
      }
      
      // The backend sends the image directly as a blob
      const imageBlob = await response.blob();
      const processedImageUrl = URL.createObjectURL(imageBlob);

      // The current backend doesn't return the noise type, so we set a generic message
      setImage({ ...image, processedUrl: processedImageUrl, noiseType: 'Detected' });
      setStatus({ isLoading: false, error: '', message: 'Image successfully denoised!' });
    } catch (err) {
      setStatus({ isLoading: false, error: err.message || 'An unknown error occurred', message: '' });
    }
  };
  
  // Handle download
  const handleDownload = (e) => {
    e.preventDefault();
    if (!image.processedUrl) return;
    const link = document.createElement('a');
    link.href = image.processedUrl;
    link.download = `denoised-xray.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Reset state
  const handleReset = (e) => {
    e.preventDefault();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setImage({ file: null, previewUrl: '', processedUrl: '', noiseType: '' });
    setStatus({ isLoading: false, error: '', message: '' });
  };

  return (
    // Section wrapper with consistent padding and ID for navigation
    <div id="Enhance" className="bg-white py-20 px-4">
      <div className="container mx-auto max-w-6xl">
        <header className="mb-12 text-center">
          <h1 className="text-4xl lg:text-5xl font-bold text-gray-900">Enhance Your X-Ray</h1>
          <p className="text-lg text-gray-600 mt-4">Upload an image to see our AI in action.</p>
        </header>

        {/* The responsive grid layout */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left panel - Upload */}
          <section className="bg-gray-50 rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">1. Upload Image</h2>
            <form onSubmit={handleSubmit}>
              <div className="mb-6">
                <input type="file" ref={fileInputRef} onChange={handleFileSelection} accept="image/jpeg,image/png,image/tiff,image/bmp" className="hidden" id="file-upload" />
                {!image.previewUrl ? (
                  <label htmlFor="file-upload" className="border-2 border-dashed border-gray-300 rounded-lg p-8 flex flex-col items-center justify-center cursor-pointer hover:border-blue-500 transition-colors">
                    <svg className="w-12 h-12 text-gray-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                    <span className="text-gray-600">Click to select an X-Ray</span>
                    <span className="text-sm text-gray-400 mt-2">JPG, PNG, TIFF, BMP</span>
                  </label>
                ) : (
                  <div className="relative"><img src={image.previewUrl} alt="Preview" className="w-full h-auto rounded-lg border" /><button type="button" onClick={handleReset} className="absolute top-2 right-2 bg-red-500 text-white w-6 h-6 flex items-center justify-center rounded-full hover:bg-red-600 text-xs">✕</button></div>
                )}
              </div>
              {image.previewUrl && (<button type="submit" disabled={status.isLoading} className={`w-full py-3 px-4 rounded-lg font-semibold text-white transition-colors ${status.isLoading ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}>{status.isLoading ? 'Processing...' : 'Denoise Image'}</button>)}
              {status.message && (<div className="mt-4 p-3 bg-green-100 text-green-800 rounded-lg text-center">{status.message}</div>)}
              {status.error && (<div className="mt-4 p-3 bg-red-100 text-red-800 rounded-lg text-center">{status.error}</div>)}
            </form>
          </section>

          {/* Right panel - Results */}
          <section className="bg-gray-50 rounded-xl shadow-lg p-6 flex flex-col">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">2. View Result</h2>
            <div className="flex-grow">
              {image.processedUrl ? (
                <div>
                  <div className="mb-4"><p className="text-gray-700 mb-2"><span className="font-medium">Result:</span> Denoised Image</p><div className="relative pt-4"><img src={image.processedUrl} alt="Denoised result" className="w-full h-auto rounded-lg border" onError={() => { setStatus({ ...status, error: 'Failed to load processed image' }); }} /><div className="absolute top-0 left-0 bg-green-100 text-green-800 rounded-tr-lg rounded-bl-lg px-2 py-1 text-xs font-medium">Denoised</div></div></div>
                  <div className="flex gap-4 mt-6">
                    <button onClick={handleDownload} className="flex-1 py-2 px-4 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-semibold">Download</button>
                    <button onClick={handleReset} className="flex-1 py-2 px-4 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors font-semibold">New Image</button>
                  </div>
                </div>
              ) : (
                <div className="bg-gray-200 rounded-lg p-8 text-center h-full flex items-center justify-center"><p className="text-gray-500">Your denoised image will appear here</p></div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default Enhance;
