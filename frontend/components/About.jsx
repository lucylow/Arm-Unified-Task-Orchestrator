import React, { useEffect, useRef, useState } from 'react';

const About = () => {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef(null);

  const skills = [
    'HTML/CSS',
    'JavaScript',
    'React',
    'Responsive Design',
    'UI/UX',
    'Git',
    'Tailwind CSS',
    'TypeScript'
  ];

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    const currentRef = sectionRef.current;

    if (currentRef) {
      observer.observe(currentRef);
    }

    return () => {
      if (currentRef) {
        observer.unobserve(currentRef);
      }
    };
  }, []);

  return (
    <section id="about" className="py-20 bg-white" ref={sectionRef}>
      <div className="container-custom">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-4 text-gray-900 relative pb-4">
          About Me
          <span className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-20 h-1 bg-primary rounded-full" />
        </h2>
        
        <div 
          className={`mt-16 grid md:grid-cols-2 gap-12 items-center transition-all duration-700 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}
        >
          {/* Text Content */}
          <div className="order-2 md:order-1">
            <h3 className="text-2xl md:text-3xl font-bold mb-6 text-gray-900">
              Creating digital experiences that matter
            </h3>
            <p className="text-gray-600 mb-4 leading-relaxed">
              I'm a passionate web developer with expertise in creating responsive, user-friendly 
              websites and applications. With a background in both design and development, I bridge 
              the gap between aesthetics and functionality.
            </p>
            <p className="text-gray-600 mb-6 leading-relaxed">
              My approach focuses on clean code, intuitive interfaces, and performance optimization 
              to deliver exceptional digital experiences across all devices.
            </p>
            
            {/* Skills Tags */}
            <div className="flex flex-wrap gap-3 mt-6">
              {skills.map((skill, index) => (
                <span
                  key={index}
                  className="px-4 py-2 bg-gray-100 rounded-full text-sm font-medium text-gray-700 hover:bg-primary hover:text-white transition-colors duration-300"
                >
                  {skill}
                </span>
              ))}
            </div>
          </div>

          {/* Image */}
          <div className="order-1 md:order-2">
            <div className="rounded-lg overflow-hidden shadow-2xl transform hover:scale-105 transition-transform duration-300">
              <img
                src="https://images.unsplash.com/photo-1494790108755-2616b612b786?ixlib=rb-1.2.1&auto=format&fit=crop&w=700&q=80"
                alt="Lucy Low"
                className="w-full h-auto object-cover"
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;

