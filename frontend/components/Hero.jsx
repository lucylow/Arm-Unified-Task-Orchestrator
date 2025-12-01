import React from 'react';

const Hero = () => {
  const handleScrollToSection = (e, href) => {
    e.preventDefault();
    const target = document.querySelector(href);
    if (target) {
      const headerOffset = 80;
      const elementPosition = target.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
      
      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      });
    }
  };

  return (
    <section 
      id="home" 
      className="pt-36 pb-24 bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50 text-center"
    >
      <div className="container-custom">
        <div className="max-w-4xl mx-auto animate-fade-in">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 text-gray-900">
            Hello, I'm Lucy Low
          </h1>
          <p className="text-xl md:text-2xl mb-8 text-gray-600">
            Web Developer & UI/UX Designer
          </p>
          <p className="text-base md:text-lg mb-10 text-gray-700 max-w-2xl mx-auto">
            I create beautiful, functional websites that provide amazing user experiences.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <a
              href="#projects"
              onClick={(e) => handleScrollToSection(e, '#projects')}
              className="w-full sm:w-auto px-8 py-3 bg-primary text-white rounded-lg font-semibold shadow-lg hover:bg-secondary hover:-translate-y-1 transition-all duration-300"
            >
              View My Work
            </a>
            <a
              href="#contact"
              onClick={(e) => handleScrollToSection(e, '#contact')}
              className="w-full sm:w-auto px-8 py-3 bg-transparent border-2 border-primary text-primary rounded-lg font-semibold hover:bg-primary hover:text-white transition-all duration-300"
            >
              Get In Touch
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;

