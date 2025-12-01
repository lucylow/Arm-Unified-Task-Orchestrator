import React from 'react';
import { Github, Linkedin, Twitter, Dribbble } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  const quickLinks = [
    { href: '#home', label: 'Home' },
    { href: '#about', label: 'About' },
    { href: '#projects', label: 'Projects' },
    { href: '#contact', label: 'Contact' }
  ];

  const services = [
    'Web Development',
    'UI/UX Design',
    'Responsive Design',
    'Website Maintenance'
  ];

  const socialLinks = [
    { icon: Github, href: 'https://github.com', label: 'GitHub' },
    { icon: Linkedin, href: 'https://linkedin.com', label: 'LinkedIn' },
    { icon: Twitter, href: 'https://twitter.com', label: 'Twitter' },
    { icon: Dribbble, href: 'https://dribbble.com', label: 'Dribbble' }
  ];

  const handleNavClick = (e, href) => {
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
    <footer className="bg-gray-900 text-white py-16">
      <div className="container-custom">
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-10 mb-12">
          {/* About Column */}
          <div>
            <h3 className="text-xl font-bold mb-4 relative pb-3">
              Lucy Low
              <span className="absolute bottom-0 left-0 w-10 h-0.5 bg-primary" />
            </h3>
            <p className="text-gray-400 mb-6 leading-relaxed">
              Creating beautiful, functional websites with a focus on user experience 
              and responsive design.
            </p>
            <div className="flex gap-3">
              {socialLinks.map((social, index) => {
                const Icon = social.icon;
                return (
                  <a
                    key={index}
                    href={social.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-10 h-10 bg-white/10 rounded-full flex items-center justify-center hover:bg-primary hover:-translate-y-1 transition-all duration-300"
                    aria-label={social.label}
                  >
                    <Icon size={18} />
                  </a>
                );
              })}
            </div>
          </div>

          {/* Quick Links Column */}
          <div>
            <h3 className="text-xl font-bold mb-4 relative pb-3">
              Quick Links
              <span className="absolute bottom-0 left-0 w-10 h-0.5 bg-primary" />
            </h3>
            <ul className="space-y-3">
              {quickLinks.map((link, index) => (
                <li key={index}>
                  <a
                    href={link.href}
                    onClick={(e) => handleNavClick(e, link.href)}
                    className="text-gray-400 hover:text-primary hover:translate-x-1 inline-block transition-all duration-300"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Services Column */}
          <div>
            <h3 className="text-xl font-bold mb-4 relative pb-3">
              Services
              <span className="absolute bottom-0 left-0 w-10 h-0.5 bg-primary" />
            </h3>
            <ul className="space-y-3">
              {services.map((service, index) => (
                <li key={index}>
                  <span className="text-gray-400">
                    {service}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Copyright */}
        <div className="pt-8 border-t border-white/10 text-center">
          <p className="text-gray-400 text-sm">
            &copy; {currentYear} Lucy Low. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

