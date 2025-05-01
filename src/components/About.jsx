import { useState, useEffect } from "react";

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

export default function RadiologyLanding() {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % carouselItems.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container mx-auto px-5 py-10 bg-gray-100 shadow-xl rounded-lg mb-20 ">
      <div className="flex flex-col  items-center justify-center">
        <div className="lg:min-w-1/2">
          <h1 className="text-4xl font-bold text-gray-900 text-center" id="op2">
            About Us
          </h1>
          <p className="text-2xl text-gray-600 mt-8 px-4">
            At X-Enhance, we are revolutionizing medical imaging with AI-powered
            noise reduction technology. Our advanced algorithms enhance X-ray
            clarity, helping physicians make more accurate diagnoses with
            improved image quality. Join us in shaping the future of medical
            imaging—where every X-ray tells a clearer story.
          </p>
        </div>

        <div className="lg:w-1/2 flex justify-center mt-8 ">
          <div className="w-full max-w-lg bg-stone-300 shadow-md rounded-lg p-6 text-center">
            <h1 className="text-2xl font-semibold text-gray-900 ">
              {carouselItems[currentIndex].title}
            </h1>
            <p className="text-gray-600 mt-2 trasntition-all ">
              {carouselItems[currentIndex].description}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}