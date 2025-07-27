import './NavBar.css';

function handleClick() {
  // This function scrolls to the element with id="Enhance"
  const enhanceSection = document.getElementById('Enhance');
  if (enhanceSection) {
    enhanceSection.scrollIntoView({ behavior: 'smooth' });
  }
}

export default function Hero() {
  return (
    // ADDED: id="home" for scrolling, the background image, and calculated height
    // The height is 100% of the viewport height (100vh) minus the navbar height (h-20 = 5rem)
    <div 
      id="home" 
      className="h-[calc(100vh-5rem)] w-full flex items-center justify-end p-4 md:p-8 bg-[url('./assets/cover.jpg')] bg-cover bg-center"
    >
      
      {/* The inner content layout remains the same as it was correct */}
      <div className="flex flex-col items-end text-right max-w-4xl">
        
        <h1 className="text-4xl md:text-6xl text-white font-medium" style={{textShadow: '2px 2px 4px rgba(0,0,0,0.7)'}}>
          We Don't Enhance <span className="text-red-500">X-Rays</span>
        </h1>
        <h1 className="text-4xl md:text-6xl text-white font-medium mt-2" style={{textShadow: '2px 2px 4px rgba(0,0,0,0.7)'}}>
          We Enhance <span className="text-red-500">Lives</span>
        </h1>

        <div className="mt-8 space-y-2" style={{textShadow: '1px 1px 3px rgba(0,0,0,0.7)'}}>
          <p className="text-xl md:text-3xl text-white font-light">
            Empowering the world with advanced Radiology AI.
          </p>
          <p className="text-xl md:text-3xl text-white font-light">
            Making the worst, better. And the better, the best.
          </p>
        </div>

        <div className="w-full flex justify-center mt-12">
          <button 
            className="rounded-full px-12 py-4 border-2 transition-all bg-gradient-to-r from-stone-100 via-stone-400 to-stone-500 font-bold hover:bg-gradient-to-r hover:from-stone-500 hover:via-stone-400 hover:to-black hover:text-white duration-300 hover:scale-105" 
            onClick={handleClick} 
          >
            ENHANCE
          </button>
        </div>
      </div>
    </div>
  );
}
