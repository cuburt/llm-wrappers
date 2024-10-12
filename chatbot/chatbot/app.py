# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:13:57 2023

@author: anilkumar.lenka

@project: XAI

@input:

@output:

@des
"""

import os
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import scripts.api.endpoint_router as endpoint_router

app = FastAPI(debug=True)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=[],
)

app.include_router(
    endpoint_router.router,
    prefix="/xairagllmiqbot",
    tags=["xaiiqbot"],
    responses={404: {"description": "Not found"}},
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', "8080")))
