import React from "react";
import { useState } from "react";
import { useEffect } from "react";
import "./App.css";
import NavBar from "./components/NavBar";
import Hero from "./components/Hero";
import Demo from "./components/Demo";
import About from "./components/About";
import Suggestions from "./components/Suggestions";
import Enhance from './components/Enhance';

function App() {
  // const [count, setCount] = useState(0);
  const handleButton = (e) => {
    console.log(e)
    console.log("Hello");
  }
  return (
     
      <div class="min-w-screen min-h-screen relative overflow-auto scroll-smooth" >
        <div class="absolute inset-0  md:bg-[url('./assets/cover.jpg')] bg-cover bg-center bg-no-repeat mt-6 overflow-x-clip" id="home">
          <NavBar />
          <Hero />
          <Demo />

          <div
            class="container bg-stone-300 absolute min-w-screen  "
            id="about"
          >
            <div class="mt-20">
              <About />
            </div>
            <div class=" flex min-w-screen min-h-screen bg-white justify-center items-center ">
              <Enhance />
              
              
            </div>
            <Suggestions />
          </div>
        </div>
      </div>
     
   
  );
}

export default App;