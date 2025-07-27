import { useState, useEffect } from "react";

// Carousel data remains the same
const carouselItems = [
  {
    title: "AI-Powered Noise Reduction",
    description:
      "Removes unwanted noise and artifacts for crystal-clear visuals.",
  },
  {
    title: "High-Resolution Enhancement",
    description: "Enhances clarity for precise medical analysis.",
  },
  {
    title: "Faster Diagnosis",
    description: "AI-driven technology speeds up medical assessments.",
  },
  {
    title: "Improved Abnormality Detection",
    description: "Helps radiologists identify fractures, tumors, and more.",
  },
];

export default function About() {
  const [currentIndex, setCurrentIndex] = useState(0);

  // Carousel effect logic remains the same
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % carouselItems.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    // Section wrapper with consistent padding and ID for navigation
    <div id="about" className="bg-gray-100 py-20 px-4">
      <div className="container mx-auto">
        {/* Main flex container: stacks on mobile, row on large screens */}
        <div className="flex flex-col lg:flex-row items-center justify-center gap-12 lg:gap-16">
          
          {/* Left Side: About Us Text */}
          <div className="lg:w-1/2 text-center lg:text-left">
            <h1 className="text-4xl lg:text-5xl font-bold text-gray-900">
              About Us
            </h1>
            <p className="text-lg text-gray-700 mt-6 max-w-xl mx-auto lg:mx-0">
              At X-Enhance, we are revolutionizing medical imaging with AI-powered
              noise reduction technology. Our advanced algorithms enhance X-ray
              clarity, helping physicians make more accurate diagnoses with
              improved image quality. Join us in shaping the future of medical
              imaging—where every X-ray tells a clearer story.
            </p>
          </div>

          {/* Right Side: Features Carousel */}
          <div className="lg:w-1/2 flex justify-center w-full">
            <div className="w-full max-w-md bg-stone-300 shadow-xl rounded-lg p-8 text-center transition-all duration-500">
              <h2 className="text-2xl font-semibold text-gray-900 h-16">
                {carouselItems[currentIndex].title}
              </h2>
              <p className="text-gray-700 mt-2 h-12">
                {carouselItems[currentIndex].description}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
