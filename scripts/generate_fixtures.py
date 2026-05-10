"""Generate PDF fixtures for eval harness.

5 synthetic CV profiles as string constants → PDF via reportlab.
NO LLM calls — fully reproducible, deterministic fixtures.
"""

from __future__ import annotations

import logging
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path("tests/fixtures")

CV_JUNIOR_TEXT = """Jan Novák
junior@example.cz | +420 123 456 789 | Praha

EXPERIENCE
Software Developer
Acme Corp, Praha
2024 - Present
Implemented REST APIs in Python, worked with PostgreSQL and Redis.
Fixed bugs in legacy Django application.

EDUCATION
BSc Computer Science
CVUT FIT, Praha, 2024

SKILLS
Python, SQL, Git, Docker, FastAPI, PostgreSQL

LANGUAGES
Czech (native), English (B2)
"""

CV_MEDIOR_TEXT = """Petra Svobodová
medior@example.cz | +420 234 567 890 | Brno

EXPERIENCE
Backend Developer
TechCorp s.r.o., Brno
2022 - Present
Developed and maintained REST APIs serving 50k+ users daily.
Led migration from Django monolith to FastAPI microservices.
Mentored 2 junior developers.

Software Developer
StartupXY, Brno
2020 - 2022
Built e-commerce backend in Python/Django.
Implemented payment integration with Stripe API.
Set up CI/CD pipeline using GitHub Actions and Docker.

EDUCATION
MSc Software Engineering
Masarykova Universita, Brno, 2020

SKILLS
Python, FastAPI, Django, PostgreSQL, Redis, Docker, Kubernetes,
Git, GitHub Actions, REST API, SQL, Linux, Pytest

LANGUAGES
Czech (native), English (C1), German (A2)
"""

CV_SENIOR_TEXT = """Tomáš Kratochvíl
senior@example.cz | +420 345 678 901 | Praha

PROFILE
Senior AI/ML Engineer with 8 years of experience building data-intensive
applications and machine learning systems. Specializing in LLM applications,
RAG architectures, and production ML deployment.

EXPERIENCE
Senior AI/ML Engineer
BigTech CZ, Praha
2022 - Present
Architected and deployed RAG system processing 1M+ documents using LLM APIs.
Built embedding pipeline with ChromaDB and Anthropic Claude.
Reduced inference latency by 60% through batching and caching strategies.
Tech lead for team of 5 engineers.

ML Engineer
DataSolutions, Praha
2019 - 2022
Developed recommendation engine for e-commerce platform (2M users).
Built training pipelines using PyTorch and TensorFlow.
Deployed models to production using Kubernetes and MLflow.

Backend Developer
WebAgency s.r.o., Praha
2016 - 2019
Developed Python/Django backend for SaaS CRM product.
Migrated from monolith to microservices architecture.

EDUCATION
MSc Artificial Intelligence
Czech Technical University, Praha, 2016

SKILLS
Python, PyTorch, TensorFlow, LLM, Embedding, RAG, ChromaDB, Anthropic, OpenAI,
FastAPI, PostgreSQL, Redis, Docker, Kubernetes, MLflow, Git, SQL, Linux,
Transformer, BERT, GPT

LANGUAGES
Czech (native), English (C2)
"""

CV_PRINCIPAL_TEXT = """Radek Novotný, PhD
principal@example.cz | +420 456 789 012 | Praha

PROFILE
Principal Software Engineer (individual contributor track) with 15 years of
hands-on experience building distributed systems and high-throughput
backend platforms. Mentors senior engineers; no direct reports.
PhD in Computer Science from Charles University.

EXPERIENCE
Principal Software Engineer
GlobalTech Praha, Praha
2019 - Present
Designed and implemented event-streaming pipeline serving 10M events/day.
Owns architecture for the core trading platform; writes production code daily.
Mentors 4 senior engineers via 1:1 technical reviews — no people-management.
Author of internal RFC process; reviews all cross-team design proposals.

Staff Software Engineer
CzechUnicorn, Praha
2016 - 2019
Built distributed event streaming platform from prototype to 10M events/day.
Hands-on platform contributor: 80% IC code, 20% architectural review.
Mentored teammates; line-management was a separate Engineering Manager role.

Senior Software Engineer
IBMCzech, Praha
2013 - 2016
Developed distributed caching layer for enterprise middleware platform.
Contributed to open-source Apache Kafka integration.

Software Engineer
Oracle Research Lab, Praha
2010 - 2013
PhD research on distributed consensus algorithms.
Published 3 peer-reviewed papers on Byzantine fault tolerance.

EDUCATION
PhD Computer Science
Karlova Universita, Praha, 2013

MSc Computer Science
Karlova Universita, Praha, 2009

SKILLS
Python, Go, Java, Distributed Systems, Kafka, Kubernetes, Docker, PostgreSQL,
Redis, Cassandra, Cloud Architecture, AWS, GCP, Terraform, CI/CD, Git,
System Design, Performance Engineering, Security Architecture, Data Engineering

LANGUAGES
Czech (native), English (C2), German (B2)
"""

CV_CONTRACTOR_TEXT = """Monika Blahová
contractor@example.cz | +420 567 890 123 | Praha (Remote available)

PROFILE
Freelance Senior Backend Developer with 10 years of experience delivering
high-quality software solutions to clients across Czech Republic and EU.
OSVČ / Self-employed since 2014.

EXPERIENCE
Freelance Backend Developer (Contract)
Client: FinTech Startup, Praha
2023 - Present
Building payment processing API in Python/FastAPI.
PostgreSQL database design and optimization.

Freelance Backend Developer (Contract)
Client: E-commerce Platform, Remote
2022 - 2023
Developed product catalog microservice handling 100k+ products.
Redis caching implementation, reduced API response time by 70%.

Freelance Backend Developer (Contract)
Client: Healthcare SaaS, Brno
2021 - 2022
GDPR-compliant patient data management system.
REST API integration with 3rd party EHR systems.

Freelance Full-stack Developer (Contract)
Client: Czech Media Group, Praha
2020 - 2021
Content management system for news portal with 500k monthly visitors.
React frontend + Django REST backend.

Freelance Backend Developer (Contract)
Client: LogisticsCo, Praha
2019 - 2020
Fleet management API and real-time tracking system.
WebSocket implementation for live driver location updates.

Software Developer
TechStart s.r.o., Praha
2014 - 2019
Full-stack development for B2B SaaS platform.
Eventually became technical lead for backend team.

EDUCATION
BSc Software Engineering
VUT FIT, Brno, 2014

SKILLS
Python, FastAPI, Django, PostgreSQL, Redis, Docker, React, JavaScript,
TypeScript, Git, REST API, WebSocket, SQL, Linux, Nginx

LANGUAGES
Czech (native), English (C1)
"""


def _text_to_pdf(text: str, output_path: Path) -> None:
    """Convert CV text string to PDF using reportlab."""
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4

    c.setFont("Helvetica", 10)
    margin = 50
    y = height - margin
    line_height = 14

    for line in text.strip().split("\n"):
        if y < margin:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - margin

        # Bold for lines that look like headers (all caps or short lines at start)
        if line.isupper() and len(line) < 30:
            c.setFont("Helvetica-Bold", 10)
        else:
            c.setFont("Helvetica", 10)

        c.drawString(margin, y, line)
        y -= line_height

    c.save()


def generate_all_fixtures() -> None:
    """Generate all 5 fixture PDFs in tests/fixtures/."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    fixtures = {
        "cv_junior.pdf": CV_JUNIOR_TEXT,
        "cv_medior.pdf": CV_MEDIOR_TEXT,
        "cv_senior.pdf": CV_SENIOR_TEXT,
        "cv_principal.pdf": CV_PRINCIPAL_TEXT,
        "cv_contractor.pdf": CV_CONTRACTOR_TEXT,
    }

    for filename, text in fixtures.items():
        output_path = FIXTURES_DIR / filename
        _text_to_pdf(text, output_path)
        print(f"Generated: {output_path} ({output_path.stat().st_size} bytes)")  # noqa: T201

    print(f"\nAll {len(fixtures)} fixtures generated in {FIXTURES_DIR}/")  # noqa: T201


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_fixtures()
