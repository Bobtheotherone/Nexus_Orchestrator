# Project Alpha

## Goals

- Build a real-time data processing pipeline
- Support at least 10,000 events per second throughput
- Provide sub-second latency for critical alerts

## Scope

- The system must ingest data from Kafka topics
- The system shall support JSON and Avro message formats
- Must provide a REST API for query access
- Should include a dashboard for monitoring pipeline health

## Constraints

- Must run on Kubernetes with max 8GB memory per pod
- All data must be encrypted at rest and in transit
- Must comply with GDPR data retention policies
- System should support horizontal scaling

## Non-Goals

- Real-time ML model training (batch only)
- Custom visualization beyond standard dashboards
