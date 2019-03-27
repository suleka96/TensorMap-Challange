import { TestBed } from '@angular/core/testing';

import { NnConfigService } from './nn-config.service';

describe('NnConfigService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: NnConfigService = TestBed.get(NnConfigService);
    expect(service).toBeTruthy();
  });
});
