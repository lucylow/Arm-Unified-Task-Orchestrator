import React, { useState, useEffect, useRef } from 'react';
import { Mail, Phone, MapPin, Github, Linkedin, Twitter, Dribbble } from 'lucide-react';

const Contact = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const sectionRef = useRef(null);

  const contactInfo = [
    {
      icon: Mail,
      title: 'Email',
      value: 'hello@lucylow.com',
      href: 'mailto:hello@lucylow.com'
    },
    {
      icon: Phone,
      title: 'Phone',
      value: '+1 (123) 456-7890',
      href: 'tel:+11234567890'
    },
    {
      icon: MapPin,
      title: 'Location',
      value: 'San Francisco, CA',
      href: '#'
    }
  ];

  const socialLinks = [
    { icon: Github, href: 'https://github.com', label: 'GitHub' },
    { icon: Linkedin, href: 'https://linkedin.com', label: 'LinkedIn' },
    { icon: Twitter, href: 'https://twitter.com', label: 'Twitter' },
    { icon: Dribbble, href: 'https://dribbble.com', label: 'Dribbble' }
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

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    alert('Thank you for your message! I will get back to you soon.');
    setFormData({
      name: '',
      email: '',
      subject: '',
      message: ''
    });
  };

  return (
    <section id="contact" className="py-20 bg-white" ref={sectionRef}>
      <div className="container-custom">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-4 text-gray-900 relative pb-4">
          Get In Touch
          <span className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-20 h-1 bg-primary rounded-full" />
        </h2>

        <div 
          className={`mt-16 grid md:grid-cols-2 gap-12 transition-all duration-700 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}
        >
          {/* Contact Info */}
          <div className="space-y-6">
            {contactInfo.map((item, index) => {
              const Icon = item.icon;
              return (
                <div key={index} className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-primary rounded-full flex items-center justify-center text-white flex-shrink-0">
                    <Icon size={20} />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900 text-lg">
                      {item.title}
                    </h3>
                    <a 
                      href={item.href}
                      className="text-gray-600 hover:text-primary transition-colors"
                    >
                      {item.value}
                    </a>
                  </div>
                </div>
              );
            })}

            {/* Social Links */}
            <div className="pt-6">
              <h3 className="font-semibold text-gray-900 text-lg mb-4">Follow Me</h3>
              <div className="flex gap-4">
                {socialLinks.map((social, index) => {
                  const Icon = social.icon;
                  return (
                    <a
                      key={index}
                      href={social.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center text-gray-700 hover:bg-primary hover:text-white hover:-translate-y-1 transition-all duration-300"
                      aria-label={social.label}
                    >
                      <Icon size={18} />
                    </a>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Contact Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label htmlFor="name" className="block font-semibold mb-2 text-gray-900">
                Name
              </label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                placeholder="Your Name"
                required
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all"
              />
            </div>

            <div>
              <label htmlFor="email" className="block font-semibold mb-2 text-gray-900">
                Email
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="Your Email"
                required
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all"
              />
            </div>

            <div>
              <label htmlFor="subject" className="block font-semibold mb-2 text-gray-900">
                Subject
              </label>
              <input
                type="text"
                id="subject"
                name="subject"
                value={formData.subject}
                onChange={handleChange}
                placeholder="Subject"
                required
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all"
              />
            </div>

            <div>
              <label htmlFor="message" className="block font-semibold mb-2 text-gray-900">
                Message
              </label>
              <textarea
                id="message"
                name="message"
                value={formData.message}
                onChange={handleChange}
                placeholder="Your Message"
                required
                rows="5"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg resize-y focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all"
              />
            </div>

            <button
              type="submit"
              className="w-full px-8 py-3 bg-primary text-white rounded-lg font-semibold shadow-lg hover:bg-secondary hover:-translate-y-1 transition-all duration-300"
            >
              Send Message
            </button>
          </form>
        </div>
      </div>
    </section>
  );
};

export default Contact;

