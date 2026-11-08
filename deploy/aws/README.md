# AWS deployment plan (UNTESTED - no AWS account was available while building)

Goal: cheapest reasonable setup for a portfolio demo. **Do not leave it running**; follow Cleanup.

## Options
| Option | Monthly cost (rough, verify with the AWS Pricing Calculator) | Notes |
|---|---|---|
| A. One EC2 `t3.small` (or Lightsail) running `docker compose` with Postgres in a container | ~US$15-20 | simplest; manage backups yourself |
| B. ECS Fargate (0.5 vCPU/1 GB) + RDS `db.t4g.micro` + S3 + CloudWatch | ~US$30-45 | closest to production; RDS is the main fixed cost |
| C. If no AWS: Render / Fly.io / Railway with the same Dockerfile + managed Postgres | free-tier to ~US$10 | same container, different ops |
The expected demo workload (a few requests/s) fits Option A comfortably; B is for demonstrating the skill set.

## Option B steps (standard AWS CLI; adapt names/regions)
1. Create a least-privilege IAM role; use `aws configure sso` - never commit keys.
2. `aws ecr create-repository --repository-name sentinelml-api`
3. `aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com`
4. `docker build -t sentinelml-api . && docker tag sentinelml-api <acct>.dkr.ecr.<region>.amazonaws.com/sentinelml-api:v1 && docker push <that tag>`
5. RDS PostgreSQL in a private subnet; security group allows 5432 only from the ECS task SG. Store `DATABASE_URL` and `SECRET_KEY` in Secrets Manager.
6. Artifacts: `services.save_bundle/load_bundle` use local disk - **an S3 backend is NOT implemented**; mount EFS or add boto3 first.
7. ECS cluster + Fargate service behind an ALB (HTTPS via ACM), health check `/health`, log driver `awslogs` -> CloudWatch.
8. GitHub Actions: OIDC role -> push image -> `aws ecs update-service --force-new-deployment`.
9. Verify: `curl https://<alb>/health`, then exercise `/docs`.
10. Rollback: `aws ecs update-service --task-definition <previous revision>`.

## Cleanup
Delete the ECS service/cluster, ALB, RDS (snapshot if needed), ECR images, S3 bucket, CloudWatch log groups, and any NAT gateway / Elastic IPs (forgotten NAT gateways are a classic surprise bill).
