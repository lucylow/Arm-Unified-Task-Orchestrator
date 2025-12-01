import React, { useEffect, useRef, useState } from 'react';
import { ExternalLink, Github } from 'lucide-react';

const Projects = () => {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef(null);

  const projects = [
    {
      id: 1,
      title: 'E-commerce Platform',
      description: 'A fully responsive e-commerce website with product filtering, cart functionality, and secure checkout.',
      image: 'https://images.unsplash.com/photo-1551650975-87deedd944c3?ixlib=rb-1.2.1&auto=format&fit=crop&w=700&q=80',
      tags: ['React', 'Node.js', 'MongoDB'],
      liveUrl: '#',
      githubUrl: '#'
    },
    {
      id: 2,
      title: 'Task Management App',
      description: 'A productivity application for organizing tasks, setting deadlines, and tracking progress.',
      image: 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-1.2.1&auto=format&fit=crop&w=700&q=80',
      tags: ['Vue.js', 'Firebase', 'PWA'],
      liveUrl: '#',
      githubUrl: '#'
    },
    {
      id: 3,
      title: 'Portfolio Website',
      description: 'A responsive portfolio website for a creative professional with project showcases and contact form.',
      image: 'https://images.unsplash.com/photo-1551434678-e076c223a692?ixlib=rb-1.2.1&auto=format&fit=crop&w=700&q=80',
      tags: ['HTML/CSS', 'JavaScript', 'GSAP'],
      liveUrl: '#',
      githubUrl: '#'
    }
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
    <section id="projects" className="py-20 bg-gray-50" ref={sectionRef}>
      <div className="container-custom">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-4 text-gray-900 relative pb-4">
          My Projects
          <span className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-20 h-1 bg-primary rounded-full" />
        </h2>

        <div className="mt-16 grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <div
              key={project.id}
              className={`bg-white rounded-lg overflow-hidden shadow-lg hover:shadow-2xl hover:-translate-y-2 transition-all duration-300 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
              }`}
              style={{ transitionDelay: `${index * 100}ms` }}
            >
              {/* Project Image */}
              <div className="h-48 overflow-hidden">
                <img
                  src={project.image}
                  alt={project.title}
                  className="w-full h-full object-cover hover:scale-110 transition-transform duration-300"
                />
              </div>

              {/* Project Content */}
              <div className="p-6">
                <h3 className="text-xl font-bold mb-3 text-gray-900">
                  {project.title}
                </h3>
                <p className="text-gray-600 mb-4 text-sm leading-relaxed">
                  {project.description}
                </p>

                {/* Tags */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {project.tags.map((tag, tagIndex) => (
                    <span
                      key={tagIndex}
                      className="px-3 py-1 bg-gray-100 rounded text-xs font-medium text-gray-600"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                {/* Links */}
                <div className="flex gap-3">
                  <a
                    href={project.liveUrl}
                    className="flex-1 px-4 py-2 bg-primary text-white rounded-lg font-semibold text-sm text-center hover:bg-secondary transition-colors duration-300 flex items-center justify-center gap-2"
                  >
                    <ExternalLink size={16} />
                    Live Demo
                  </a>
                  <a
                    href={project.githubUrl}
                    className="flex-1 px-4 py-2 border-2 border-primary text-primary rounded-lg font-semibold text-sm text-center hover:bg-primary hover:text-white transition-all duration-300 flex items-center justify-center gap-2"
                  >
                    <Github size={16} />
                    Code
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Projects;

